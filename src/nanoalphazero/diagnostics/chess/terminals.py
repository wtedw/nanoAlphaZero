"""Bounded terminal-value diagnostics; no tracker, replay, or training RNG access."""
import time

import jax
import jax.numpy as jnp
import numpy as np

from nanoalphazero.diagnostics.chess.cases import terminal_cases
from nanoalphazero.mcts import make_muzero_mcts


def terminal_metrics(outcome, value_only, predictions):
    """Reduce predictions using player-relative targets and root-relative Qs.

    No root-value error: a draw-producing action need not be optimal at its root.
    The observed terminal encoding is diagnostic only, not a sampled train root.
    """
    p = {k: np.asarray(v) for k, v in predictions.items()}
    terminal_target = -outcome if value_only else np.zeros_like(outcome)
    next_target = -terminal_target
    q1 = p['reward'] - p['terminal_value']
    q2 = p['reward'] - p['greedy_reward'] + p['greedy_value']
    q3 = p['reward'] - p['greedy_reward'] + p['greedy2_reward'] - p['greedy2_value']
    metrics = {k: p[k] for k in ('root_value', 'observed_terminal_value', 'terminal_value',
                                'reward', 'greedy_value', 'greedy_reward', 'greedy2_value',
                                'greedy2_reward', 'terminal_policy_entropy')}
    metrics.update(
        terminal_value_error=np.abs(p['terminal_value'] - terminal_target),
        reward_error=np.abs(p['reward'] - (0 if value_only else outcome)),
        greedy_value_error=np.abs(p['greedy_value'] - next_target),
        greedy2_value_error=np.abs(p['greedy2_value'] - terminal_target),
        q_one_step=q1, q_two_step=q2, q_three_step=q3,
        q_one_step_error=np.abs(q1 - outcome), q_two_step_error=np.abs(q2 - outcome),
        q_three_step_error=np.abs(q3 - outcome))
    for name in ('uniform', 'top_policy'):
        values, rewards = p[name + '_value'], p[name + '_reward']
        metrics[name + '_value_mae'] = np.mean(np.abs(values - next_target[None]), axis=0)
        metrics[name + '_q_mae'] = np.mean(np.abs(p['reward'][None] - rewards + values - outcome[None]), axis=0)
    return metrics


def make_chess_terminal_probe(model, env, config, *, uniform_actions=32, top_actions=8):
    """Return probe(params) -> metrics, tables. Fixed suite/keys; bounded batches.

    Every sampled action branches from the SAME terminal latent. Greedy chains
    instead follow two successive argmax actions. These are unmasked policies.
    """
    import pgx1

    # Keep this rank-zero-only probe on a local device. A TPU-only run need
    # not enable a CPU backend, nor initialize another distributed backend.
    device = jax.local_devices()[0]
    with jax.default_device(device):
        cases = terminal_cases(pgx1.make('chess', use_bitmask=True))
        roots = jax.tree.map(lambda *xs: jnp.stack(xs), *(c[2] for c in cases))
        following = jax.tree.map(lambda *xs: jnp.stack(xs), *(c[4] for c in cases))
        observation = env.observe(roots, roots.current_player)
        after_observation = env.observe(following, following.current_player)
        outcome = np.asarray(following.rewards[jnp.arange(len(cases)), roots.current_player])
    roots, observation, after_observation = jax.device_put((roots, observation, after_observation), device)
    actions = jax.device_put(np.asarray([c[3] for c in cases], np.int32), device)
    action_count = config['game_num_actions']
    # Stable IDs across checkpoints, sampled without replacement per case.
    rng = np.random.default_rng(0)
    uniform = jax.device_put(np.stack([rng.choice(action_count, uniform_actions, replace=False)
                                      for _ in cases], axis=1).astype(np.int32), device)
    search = make_muzero_mcts(config, env, model, return_diagnostics=True)

    @jax.jit
    def predict(params):
        def initial(obs):
            return model.apply({'params': params}, obs, method=model.initial)

        def recurrent(latent, action):
            return model.apply({'params': params}, latent, action, method=model.recurrent)

        latent, _, root_value = initial(observation)
        _, _, observed = initial(after_observation)
        terminal, reward, logits, value = recurrent(latent, actions)
        next_latent, r2, next_logits, v2 = recurrent(terminal, jnp.argmax(logits, axis=-1))
        _, r3, _, v3 = recurrent(next_latent, jnp.argmax(next_logits, axis=-1))

        def branch(_, action):
            _, r, _, v = recurrent(terminal, action)
            return None, (r, v)

        # Scan small case batches instead of materializing 32 copies of latents.
        _, (ur, uv) = jax.lax.scan(branch, None, uniform)
        top = jax.lax.top_k(logits, top_actions)[1].T
        _, (tr, tv) = jax.lax.scan(branch, None, top)
        log_pi = jax.nn.log_softmax(logits)
        return dict(root_value=root_value, observed_terminal_value=observed,
                    terminal_value=value, reward=reward, greedy_value=v2, greedy_reward=r2,
                    greedy2_value=v3, greedy2_reward=r3, uniform_value=uv, uniform_reward=ur,
                    top_policy_value=tv, top_policy_reward=tr,
                    terminal_policy_entropy=-jnp.sum(jnp.exp(log_pi) * log_pi, axis=-1))

    def probe(params):
        start = time.monotonic()
        params = jax.device_put(params, device)
        predictions = jax.device_get(predict(params))
        if not all(np.isfinite(x).all() for x in predictions.values()):
            raise RuntimeError('Nonfinite chess terminal predictions')
        per_case = terminal_metrics(outcome, config.get('muzero_predict_value_only', False), predictions)
        # Deterministic root selection noise, independent of training RNG.
        result = jax.device_get(search(jax.random.PRNGKey(0), roots, params, 0., len(cases)))
        ids = np.asarray(actions)
        visits = np.asarray(result.visit_counts)[np.arange(len(cases)), ids]
        q = np.asarray(result.diagnostics['backed_up_q'])[np.arange(len(cases)), ids]
        per_case['search_visits'] = visits
        per_case['search_selected'] = (np.asarray(result.action) == ids).astype(float)
        metrics, rows = {}, []
        for i, (name, fen, _, action, _, _) in enumerate(cases):
            for key, values in per_case.items():
                metrics[f'chess/terminal/{name}/{key}'] = float(values[i])
            # Never report an unvisited action's placeholder zero as a Q estimate.
            if visits[i] > 0:
                metrics[f'chess/terminal/{name}/search_q'] = float(q[i])
            rows.append([name, fen, action, float(outcome[i]),
                         *[float(v[i]) for v in per_case.values()], float(q[i]) if visits[i] else None])
        for key in ('terminal_value_error', 'q_one_step_error', 'q_two_step_error',
                    'q_three_step_error', 'uniform_value_mae', 'top_policy_value_mae'):
            metrics[f'chess/terminal/mean/{key}'] = float(np.mean(per_case[key]))
        metrics['chess/terminal/mate_selected_fraction'] = float(np.mean(per_case['search_selected'][outcome > 0]))
        metrics['timing/chess_terminal_seconds'] = time.monotonic() - start
        tables = {'chess/terminal': dict(columns=['case', 'fen', 'action', 'outcome',
                                                *per_case, 'search_q'], data=rows)}
        return metrics, tables

    return probe
