"""MuZero opening-head diagnostics, independent of training RNG and replay.

Ports the historical standalone head tables and board charts. Values after an
opening use the next player's (White's) perspective, matching solved Hex values.
"""

import numpy as np


OPENING_VALUE_COLUMNS = [
    "game", "boardsize", "action", "row", "col", "value", "ground_truth", "highlight", "label",
]


def board_table(values, ground_truth=None, highlights=()):
    values = np.asarray(values)
    size = values.shape[0]
    highlights = set(map(int, highlights))
    rows = [["hex", size, r * size + c, r, c, float(values[r, c]),
             None if ground_truth is None else float(ground_truth[r, c]),
             r * size + c in highlights, f"r{r}c{c}"]
            for r in range(size) for c in range(size)]
    return dict(columns=OPENING_VALUE_COLUMNS, data=rows)


def board_figure(table, title, logits=False):
    import plotly.graph_objects as go
    size = table["data"][0][1]
    values = np.array([row[5] for row in table["data"]]).reshape(size, size)
    highlights = [row[2] for row in table["data"] if row[7]]
    label = "Logit" if logits else "Value"
    fig = go.Figure(go.Heatmap(
        z=values.tolist(), x=list(range(size)), y=list(range(size)),
        text=[[f"{x:+.2f}" for x in row] for row in values], texttemplate="%{text}",
        colorscale="Viridis" if logits else "RdBu", zmid=None if logits else 0.,
        colorbar=dict(title=label),
        hovertemplate=f"row=%{{y}}<br>col=%{{x}}<br>{label.lower()}=%{{z:.4f}}<extra></extra>"))
    if highlights:
        fig.add_trace(go.Scatter(
            x=[i % size for i in highlights], y=[i // size for i in highlights], mode="markers",
            marker=dict(symbol="square-open", size=34, color="red" if logits else "royalblue", line=dict(width=3)),
            name="Top-k" if logits else "Bottom-k", hoverinfo="skip"))
    fig.update_layout(title=title, xaxis=dict(title="Column", constrain="domain"),
                      yaxis=dict(title="Row", autorange="reversed", scaleanchor="x", scaleratio=1, constrain="domain"),
                      margin=dict(l=40, r=20, t=60, b=40))
    return fig



def make_muzero_head_probe(model, env, config):
    """Compile a small fixed opening suite once; return scalars and numeric tables."""
    import jax
    import jax.numpy as jnp
    from nanoalphazero.eval.hex.perfect_play import perfect_play_values

    size = config["boardsize"]
    count = size * size
    actions = jnp.arange(count)
    root = env.init_dummy_estate(count)
    following = env.step(root, actions)
    observation = env.observe(root, root.current_player)
    after_observation = env.observe(following, following.current_player)
    legal = np.asarray(following.legal_action_mask)
    truth = perfect_play_values(size)

    @jax.jit
    def predict(params):
        latent, logits, value = model.apply(
            {"params": params}, observation, method=model.initial)
        _, after_logits, after_value = model.apply(
            {"params": params}, after_observation, method=model.initial)
        _, reward, latent_logits, latent_value = model.apply(
            {"params": params}, latent, actions, method=model.recurrent)
        return logits[0], value[0], after_logits, after_value, reward, latent_logits, latent_value

    def probe(params):
        logits, initial_value, after_logits, after_value, reward, latent_logits, latent_value = (
            np.asarray(x) for x in jax.device_get(predict(params)))
        arrays = (logits, initial_value, after_logits, after_value, reward, latent_logits, latent_value)
        if not all(np.isfinite(x).all() for x in arrays):
            raise RuntimeError("Nonfinite MuZero opening-head predictions")
        tables = {
            "model/values_after_black_moves_table": board_table(
                after_value.reshape(size, size), truth, np.argsort(after_value)[:size]),
            "model/logits_heatmap_top_k_table": board_table(
                logits.reshape(size, size), highlights=np.argsort(-logits)[:size]),
            "muzero/model/latent_values_after_black_moves_table": board_table(
                latent_value.reshape(size, size), truth, np.argsort(latent_value)[:size]),
            "muzero/model/reply_policy_logits_table": dict(
                columns=["opening", "row", "col", "policy_logit", "latent_policy_logit", "legal"],
                data=[[opening, action // size, action % size,
                       float(after_logits[opening, action]), float(latent_logits[opening, action]),
                       bool(legal[opening, action])]
                      for opening in range(count) for action in range(count)]),
        }

        def entropy(values):
            probs = np.exp(values - np.max(values))
            probs /= probs.sum()
            return float(-np.sum(probs * np.log(probs + 1e-8))), probs

        policy_entropy, probabilities = entropy(logits)
        value_entropy, _ = entropy(after_value)
        metrics = {
            "model/init_board_value": float(initial_value),
            # Historical name: entropy of softmax across opening values, not
            # entropy of a categorical value head (this model has scalar values).
            "model/value_head_empty_board_entropy": value_entropy,
            "model/policy_head_empty_board_entropy": policy_entropy,
            "params-logits/init_logits_norm": float(np.linalg.norm(logits)),
            "params-logits/init_logits_std": float(np.std(logits)),
            "params-logits/init_logits_max": float(np.max(logits)),
            "params-logits/init_logits_min": float(np.min(logits)),
            "params-logits/init_logits_entropy": policy_entropy,
            "params-logits/init_logits_entropy_its": policy_entropy / np.log(2),
            "params-logits/num_moves_over_5pct": int(np.sum(probabilities > .05)),
            "muzero/model/latent_value_mse_vs_observed": float(np.mean((latent_value - after_value) ** 2)),
            "muzero/model/opening_reward_mse": float(np.mean(
                (reward - np.asarray(following.rewards)[np.arange(count), np.asarray(root.current_player)]) ** 2)),
        }
        if truth is not None:
            truth_flat = truth.reshape(-1)
            metrics.update({
                "hex_perf/mse_vs_perfect": float(np.mean((after_value - truth_flat) ** 2)),
                "hex_perf/sign_accuracy": float(np.mean(np.sign(after_value) == np.sign(truth_flat))),
                "muzero/hex_perf/latent_mse_vs_perfect": float(np.mean((latent_value - truth_flat) ** 2)),
            })
        histograms = {
            "model/opening_policy_logits_distribution": logits,
            "model/values_after_black_moves_distribution": after_value,
            "muzero/model/latent_values_after_black_moves_distribution": latent_value,
            "muzero/model/opening_predicted_rewards_distribution": reward,
        }
        return metrics, tables, histograms

    return probe


def head_diagnostic_logs(tables, histograms, backend):
    """Wrap numeric diagnostics for the selected tracker, without importing the other."""
    import importlib

    tracker = importlib.import_module(backend)
    logs = {name: tracker.Table(**table) for name, table in tables.items()}
    logs.update({name: tracker.Histogram(values) for name, values in histograms.items()})
    for table_key, chart_key, title, logits in (
        ("model/values_after_black_moves_table", "model/values_after_black_moves_heatmap",
         "Value After Each Black Opening (White perspective)", False),
        ("model/logits_heatmap_top_k_table", "model/logits_heatmap_top_k",
         "Opening Policy Logits (Top-k Highlighted)", True),
        ("muzero/model/latent_values_after_black_moves_table",
         "muzero/model/latent_values_after_black_moves_heatmap",
         "Learned-Latent Value After Each Black Opening (White perspective)", False),
    ):
        figure = board_figure(tables[table_key], title, logits)
        logs[chart_key] = tracker.Plotly(figure) if backend == "wandb" else tracker.Html(figure)
    return logs
