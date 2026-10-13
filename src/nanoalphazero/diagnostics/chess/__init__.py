"""Chess opening diagnostics and notebook charts.

Observed and imagined successor values use Black's perspective. Rewards and
root Q use White's perspective. Policy comparisons normalize over real legal
replies; illegal probability mass is reported separately in the opening table.
"""

import chess
import numpy as np

from nanoalphazero.eval.chess.action import PgxAction


SEARCH = "chess/search"
VALUES = "chess/opening_model"
REPLIES = "chess/reply_policy"
SEARCH_FIELDS = (
    "prior_logit", "gumbel", "completed_q", "transformed_q", "advantage",
    "final_score", "search_logit", "child_value", "backed_up_q",
    "policy_weight", "visits",
)


def opening_suite():
    board = chess.Board()
    return sorted((PgxAction.encode(move, board), move.uci()) for move in board.legal_moves)


def probabilities(logits, legal=None):
    logits = np.asarray(logits, dtype=np.float64)
    if legal is not None:
        logits = np.where(legal, logits, -np.inf)
    weights = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return weights / weights.sum(axis=-1, keepdims=True)


def make_chess_probe(model, env, config):
    import jax
    import jax.numpy as jnp
    from nanoalphazero.buffers import unpack_bitmask_vmap
    from nanoalphazero.mcts import make_muzero_mcts

    suite = opening_suite()
    actions = jnp.asarray([action for action, _ in suite])
    root = env.init_dummy_estate(1)
    roots = jax.tree.map(lambda x: jnp.repeat(x, len(suite), axis=0), root)
    following = env.step(roots, actions)
    observation = env.observe(roots, roots.current_player)
    after_observation = env.observe(following, following.current_player)
    legal = np.asarray(jax.device_get(
        unpack_bitmask_vmap(following.legal_action_bitmask)
        if getattr(following, "legal_action_bitmask", None) is not None
        else following.legal_action_mask))
    reward_target = np.asarray(jax.device_get(
        following.rewards[jnp.arange(len(suite)), roots.current_player]))
    search = make_muzero_mcts(config, env, model, return_diagnostics=True)

    @jax.jit
    def predict(params):
        latent, _, initial_value = model.apply({"params": params}, observation, method=model.initial)
        _, real_logits, real_value = model.apply(
            {"params": params}, after_observation, method=model.initial)
        _, reward, latent_logits, latent_value = model.apply(
            {"params": params}, latent, actions, method=model.recurrent)
        if config.get("muzero_predict_value_only", False):
            reward = jnp.zeros_like(reward)  # Effective search reward; head is unused.
        return initial_value[0], real_logits, real_value, reward, latent_logits, latent_value

    def probe(params):
        predictions = jax.device_get(predict(params))
        if not all(np.isfinite(x).all() for x in predictions):
            raise RuntimeError("Nonfinite chess opening-head predictions")
        # Fixed diagnostic noise makes checkpoints comparable without consuming
        # the training RNG. Use the configured self-play Gumbel scale.
        output = jax.device_get(search(
            jax.random.PRNGKey(0), root, params, config.get("mcts_gumbel_scale", 1.0), 1))
        initial, real_logits, real_value, reward, latent_logits, latent_value = predictions
        real_pi = probabilities(real_logits, legal)
        latent_pi = probabilities(latent_logits, legal)
        real_illegal = np.sum(probabilities(real_logits) * ~legal, axis=1)
        latent_illegal = np.sum(probabilities(latent_logits) * ~legal, axis=1)
        search_rows, value_rows, reply_rows = [], [], []
        for index, (action, move) in enumerate(suite):
            visits = float(output.visit_counts[0, action])
            fields = {name: float(output.diagnostics[name][0, action])
                      for name in SEARCH_FIELDS[:-2]}
            if not visits:
                fields["child_value"] = fields["backed_up_q"] = None
            fields.update(policy_weight=float(output.action_weights[0, action]), visits=visits)
            search_rows.append([action, move, *[fields[name] for name in SEARCH_FIELDS],
                                float(output.diagnostics["raw_value"][0]),
                                float(output.diagnostics["mixed_value"][0]),
                                bool(output.action[0] == action)])
            value_rows.append([
                action, move, float(real_value[index]), float(latent_value[index]),
                float(latent_value[index] - real_value[index]), float(reward[index]),
                float(reward_target[index]), float(real_illegal[index]), float(latent_illegal[index]),
                float(np.sum(np.abs(real_pi[index] - latent_pi[index])) / 2),
            ])
            board = chess.Board()
            board.push_uci(move)
            for reply in sorted(board.legal_moves, key=lambda m: m.uci()):
                label = PgxAction.encode(reply, board)
                if not legal[index, label]:
                    raise ValueError(f"Chess diagnostic legal-action mismatch: {move} {reply}")
                reply_rows.append([move, label, reply.uci(), float(real_logits[index, label]),
                                   float(latent_logits[index, label]), float(real_pi[index, label]),
                                   float(latent_pi[index, label])])
        tables = {
            SEARCH: dict(columns=["action", "move", *SEARCH_FIELDS, "raw_value", "mixed_value", "selected"],
                         data=search_rows),
            VALUES: dict(columns=["action", "move", "observed_value", "latent_value", "value_error",
                                  "predicted_reward", "actual_reward", "observed_illegal_mass",
                                  "latent_illegal_mass", "legal_policy_total_variation"], data=value_rows),
            REPLIES: dict(columns=["opening", "action", "reply", "observed_logit", "latent_logit",
                                   "observed_probability", "latent_probability"], data=reply_rows),
        }
        return {"model/init_board_value_chess": float(initial)}, tables

    return probe


def search_figure(table):
    import plotly.graph_objects as go

    columns, rows = table["columns"], table["data"]
    labels = [["" for _ in range(8)] for _ in range(4)]
    custom = [[None for _ in range(8)] for _ in range(4)]
    positions = []
    for row in rows:
        move = row[1]
        # Knight moves occupy a1/c1/f1/h1; pawn moves their destinations.
        square = {"b1a3": "a1", "b1c3": "c1", "g1f3": "f1", "g1h3": "h1"}.get(move, move[2:4])
        r, c = 4 - int(square[1]), ord(square[0]) - ord("a")
        positions.append((r, c))
        labels[r][c], custom[r][c] = move, row

    def grid(field):
        result = [[None for _ in range(8)] for _ in range(4)]
        for row, (r, c) in zip(rows, positions):
            result[r][c] = row[columns.index(field)]
        return result

    hover = "<br>".join(f"{name}=%{{customdata[{i}]}}" for i, name in enumerate(columns)) + "<extra></extra>"
    figure = go.Figure(go.Heatmap(
        z=grid("search_logit"), x=list("abcdefgh"), y=[4, 3, 2, 1], text=labels,
        texttemplate="%{text}", customdata=custom, hovertemplate=hover, hoverongaps=False,
        colorscale="RdBu", colorbar=dict(title="search_logit")))
    figure.update_layout(
        title="White opening search (fixed diagnostic noise)",
        xaxis=dict(title="File"), yaxis=dict(title="Rank", scaleanchor="x"),
        updatemenus=[dict(active=SEARCH_FIELDS.index("search_logit"),
                         buttons=[dict(label=field, method="restyle",
                                      args=[{"z": [grid(field)], "colorbar.title.text": field}])
                                   for field in SEARCH_FIELDS])],
    )
    return figure


def value_figure(table):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    rows = table["data"]
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           subplot_titles=("Successor value (Black perspective)", "Reward (White perspective)"))
    for column, name, panel in ((2, "Observed value", 1), (3, "Latent value", 1),
                                (5, "Predicted reward", 2), (6, "Actual reward", 2)):
        figure.add_trace(go.Scatter(x=[r[1] for r in rows], y=[r[column] for r in rows],
                                   mode="lines+markers", name=name), row=panel, col=1)
    figure.update_layout(title="Real move versus learned dynamics", height=600)
    return figure


def policy_figure(table):
    import plotly.graph_objects as go

    openings = list(dict.fromkeys(row[0] for row in table["data"]))

    def series(opening):
        rows = [row for row in table["data"] if row[0] == opening]
        return dict(x=[[r[2] for r in rows]] * 2,
                    y=[[r[5] for r in rows], [r[6] for r in rows]])

    initial = series(openings[0])
    figure = go.Figure([go.Bar(name=name, x=initial["x"][i], y=initial["y"][i])
                       for i, name in enumerate(("Observed", "Latent"))])
    figure.update_layout(
        title="Black reply policy after White's opening", barmode="group",
        yaxis_title="Probability conditioned on real legal replies",
        updatemenus=[dict(buttons=[dict(label=opening, method="restyle", args=[series(opening)])
                                   for opening in openings])],
    )
    return figure


def chess_figures(tables):
    """Render available probe tables; usable directly from a notebook."""
    return {name: render(tables[name])
            for name, render in ((SEARCH, search_figure), (VALUES, value_figure), (REPLIES, policy_figure))
            if name in tables}
