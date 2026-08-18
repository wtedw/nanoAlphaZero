"""Built-in training configurations for supported games."""


# =============================================================================
# Configuration
# =============================================================================
def get_ttt_config():
    board_size = 3
    game_max_steps = board_size * board_size
    batch_size = 4096
    REPLAY_BUFFER_TOTAL_SIZE = 1_024_000

    selfplay_buffer_len = game_max_steps + 10
    replay_buffer_len = REPLAY_BUFFER_TOTAL_SIZE // batch_size
    buffer_warmup_steps = (selfplay_buffer_len + replay_buffer_len) * 1

    return {
        # --- Game ---
        "env_id": "tic_tac_toe",
        "game_max_steps": game_max_steps,
        "num_exploratory_moves": 4,
        "env_forbids_draws": False,
        "env_allows_draws": True,
        "boardsize": board_size,
        # game_obs_shape and game_num_actions are derived from the live env in make_alphazero
        "game_obs_shape": None,
        "game_num_actions": board_size * board_size,
        # --- Model ---
        "conv_width": 32,
        "conv_depth": 4,
        # --- MCTS ---
        "mcts_num_simulations": 13,
        "mcts_variant": "1sh",
        "mcts_max_m": 9,
        "mcts_num_root_considered": 9,
        "mcts_num_survivors": 4,
        "mcts_num_k_actions": 9,
        "mcts_use_gumbel": True,
        "mcts_gumbel_scale": 1.0,
        "mcts_epsilon": 1e-8,
        "mcts_rescale_values": False,
        "mcts_value_scale": 1.0,
        "mcts_use_mixed_value": True,
        "mcts_maxvisit_init": 50,
        "mcts_bnk_rehydrate_fields": False,  # [todo] not implemented
        # --- Training ---
        "num_iters": 10_000,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "weight_decay_kernels_only": True,
        "use_bf16": False,
        "lr_warmup_steps": buffer_warmup_steps,
        "train_batch_size": batch_size,
        "cycle_n_selfplay": 10,
        "cycle_n_train": 10,
        # --- Self-play & Buffers ---
        "selfplay_batch_size": batch_size,
        "selfplay_buffer_add_batch_size": batch_size,
        "selfplay_buffer_sample_batch_size": batch_size,
        "selfplay_buffer_min_len": game_max_steps,
        "selfplay_buffer_max_len": game_max_steps,
        "selfplay_buffer_consume_size": batch_size,
        "replay_buffer_total_size": REPLAY_BUFFER_TOTAL_SIZE,
        "replay_buffer_add_batch_size": batch_size,
        "replay_buffer_sample_batch_size": batch_size,
        "replay_buffer_min_len": 1,
        "replay_buffer_max_len": replay_buffer_len,
        "replay_buffer_warmup_steps": buffer_warmup_steps,
        # --- Diagnostics & strength eval ---
        "diagnostic_period": 100,
        "eval_period": 50,  # run every N cycles
        "eval_max_plies": None,
        "ckpt_period": None,  # save a checkpoint every N cycles (None = only at end)
        # --- System ---
        "enable_sharding": True,
    }


def get_hex_config(board_size=4):
    board_cfgs = {
        4: dict(
            conv_width=64,
            conv_depth=4,
            num_iters=(500 * 10),
            learning_rate=1e-3,
            cycle_n_selfplay=10,
            cycle_n_train=10,
        ),
        5: dict(
            conv_width=128,
            conv_depth=4,
            num_iters=(1700 * 20),
            learning_rate=1e-4,
            cycle_n_selfplay=20,
            cycle_n_train=20,
        ),
        6: dict(
            conv_width=256,
            conv_depth=8,
            num_iters=(3500 * 30),
            learning_rate=1e-4,
            cycle_n_selfplay=30,
            cycle_n_train=30,
        ),
        7: dict(
            conv_width=256,
            conv_depth=16,
            num_iters=(5000 * 40),
            learning_rate=1e-4,
            cycle_n_selfplay=40,
            cycle_n_train=30,
        ),
        8: dict(
            conv_width=256,
            conv_depth=16,
            num_iters=(10_000 * 60),
            learning_rate=1e-4,
            cycle_n_selfplay=60,
            cycle_n_train=50,
        ),
        9: dict(
            conv_width=256,
            conv_depth=32,
            num_iters=(40_000 * 80),
            learning_rate=1e-4,
            cycle_n_selfplay=40,
            cycle_n_train=25,
        ),
    }
    if board_size not in board_cfgs:
        raise ValueError(f"Unsupported hex board size: {board_size}")
    board_cfg = board_cfgs[board_size]

    game_max_steps = board_size * board_size

    # ---- shrunk for easy single-file runs (hex prod = 8192) ----
    batch_size = 8192
    REPLAY_BUFFER_TOTAL_SIZE = 2_048_000
    # ------------------------------------------------------------

    selfplay_buffer_len = game_max_steps + 20
    replay_buffer_len = REPLAY_BUFFER_TOTAL_SIZE // batch_size
    buffer_warmup_steps = (selfplay_buffer_len + replay_buffer_len) * 1

    return {
        # --- Game ---
        "env_id": f"hexnoswap_{board_size}x{board_size}",
        "game_max_steps": game_max_steps,
        "num_exploratory_moves": game_max_steps // 2,
        # hex has no draws: the last player to move always wins.
        "env_forbids_draws": True,
        "env_allows_draws": False,
        "boardsize": board_size,
        "game_obs_shape": None,
        "game_num_actions": None,  # patched from live env in make_alphazero
        # --- Model ---
        "conv_width": board_cfg["conv_width"],
        "conv_depth": board_cfg["conv_depth"],
        # --- MCTS (1sh) ---
        "mcts_num_simulations": 24,
        "mcts_variant": "1sh",
        "mcts_max_m": 16,
        "mcts_num_root_considered": 16,
        "mcts_num_survivors": 8,
        "mcts_num_k_actions": game_max_steps,  # = board_size**2
        "mcts_use_gumbel": True,
        "mcts_gumbel_scale": 1.0,
        "mcts_epsilon": 1e-8,
        "mcts_rescale_values": False,
        "mcts_value_scale": 1.0,
        "mcts_use_mixed_value": True,
        "mcts_maxvisit_init": 50,
        "mcts_bnk_rehydrate_fields": False,
        # bnk off for hex (full (A,) policy targets); root temperature from prod hex
        "exp_bnk_action_weights": False,
        "exp_use_root_temperature": True,
        "exp_root_temperature": 1.3,
        # --- Training ---
        "num_iters": board_cfg["num_iters"],
        "learning_rate": board_cfg["learning_rate"],
        "weight_decay": 1e-4,
        "weight_decay_kernels_only": True,
        "use_bf16": False,
        "lr_warmup_steps": buffer_warmup_steps,
        "train_batch_size": batch_size,
        "cycle_n_selfplay": board_cfg["cycle_n_selfplay"],
        "cycle_n_train": board_cfg["cycle_n_train"],
        # --- Self-play & Buffers ---
        "selfplay_batch_size": batch_size,
        "selfplay_buffer_add_batch_size": batch_size,
        "selfplay_buffer_sample_batch_size": batch_size,
        "selfplay_buffer_min_len": selfplay_buffer_len,
        "selfplay_buffer_max_len": selfplay_buffer_len,
        "selfplay_buffer_consume_size": batch_size,
        "replay_buffer_total_size": REPLAY_BUFFER_TOTAL_SIZE,
        "replay_buffer_add_batch_size": batch_size,
        "replay_buffer_sample_batch_size": batch_size,
        "replay_buffer_min_len": 1,
        "replay_buffer_max_len": replay_buffer_len,
        "replay_buffer_warmup_steps": buffer_warmup_steps,
        # --- Diagnostics & strength eval ---
        "diagnostic_period": 50,
        "eval_period": 50,  # run every N cycles
        "eval_max_plies": None,
        "eval_opening_plies": 2,
        "ckpt_period": None,  # save a checkpoint every N cycles (None = only at end)
        # --- System ---
        "enable_sharding": True,
    }


def get_connect4_config():
    game_max_steps = 42  # 6 rows x 7 cols
    batch_size = 8192
    REPLAY_BUFFER_TOTAL_SIZE = 2_048_000

    selfplay_buffer_len = game_max_steps + 10
    replay_buffer_len = REPLAY_BUFFER_TOTAL_SIZE // batch_size
    buffer_warmup_steps = (selfplay_buffer_len + replay_buffer_len) * 1

    return {
        # --- Game ---
        "env_id": "connect_four",
        "game_max_steps": game_max_steps,
        "num_exploratory_moves": 21,
        # connect4 can end in a draw (full board), so draws are allowed.
        "env_forbids_draws": False,
        "env_allows_draws": True,
        "boardsize": 7,  # number of columns (= action space); board is 6x7
        "game_obs_shape": None,
        "game_num_actions": 7,
        # --- Model ---
        "conv_width": 128,
        "conv_depth": 8,
        # --- MCTS (1sh) ---
        "mcts_num_simulations": 64,
        "mcts_variant": "1sh",
        "mcts_max_m": 7,
        "mcts_num_root_considered": 7,
        "mcts_num_survivors": 3,
        "mcts_num_k_actions": 7,
        "mcts_use_gumbel": True,
        "mcts_gumbel_scale": 1.0,
        "mcts_epsilon": 1e-8,
        "mcts_rescale_values": False,
        "mcts_value_scale": 1.0,
        "mcts_use_mixed_value": True,
        "mcts_maxvisit_init": 50,
        "mcts_bnk_rehydrate_fields": False,
        "exp_bnk_action_weights": False,
        # --- Training ---
        "num_iters": 2100 * 20,
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "weight_decay_kernels_only": True,
        "use_bf16": False,
        "lr_warmup_steps": buffer_warmup_steps,
        "train_batch_size": batch_size,
        "cycle_n_selfplay": 20,
        "cycle_n_train": 12,
        # --- Self-play & Buffers ---
        "selfplay_batch_size": batch_size,
        "selfplay_buffer_add_batch_size": batch_size,
        "selfplay_buffer_sample_batch_size": batch_size,
        "selfplay_buffer_min_len": selfplay_buffer_len,
        "selfplay_buffer_max_len": selfplay_buffer_len,
        "selfplay_buffer_consume_size": batch_size,
        "replay_buffer_total_size": REPLAY_BUFFER_TOTAL_SIZE,
        "replay_buffer_add_batch_size": batch_size,
        "replay_buffer_sample_batch_size": batch_size,
        "replay_buffer_min_len": 1,
        "replay_buffer_max_len": replay_buffer_len,
        "replay_buffer_warmup_steps": buffer_warmup_steps,
        # --- Diagnostics & strength eval ---
        "diagnostic_period": 50,
        "eval_period": 50,  # run every N cycles
        "eval_max_plies": None,
        "eval_opening_plies": 3,
        "ckpt_period": 700,  # save a checkpoint every N cycles (None = only at end)
        # --- System ---
        "enable_sharding": True,
    }


def get_chess_config():
    board_size = 8
    GAME_MAX_STEPS = 512

    selfplay_bs = 8192
    train_bs = 8192
    REPLAY_BUFFER_TOTAL_SIZE = 4096000 * 2
    # -----------------------------------------------------------------------

    selfplay_buffer_len = GAME_MAX_STEPS + 20
    replay_buffer_len = REPLAY_BUFFER_TOTAL_SIZE // train_bs
    buffer_warmup_steps = selfplay_buffer_len + replay_buffer_len

    return {
        # --- Game ---
        "env_id": "chess",
        "game_max_steps": GAME_MAX_STEPS,
        "num_exploratory_moves": 30,
        "env_forbids_draws": False,
        "env_allows_draws": True,
        "boardsize": board_size,
        "game_obs_shape": None,
        "game_num_actions": None,  # patched from live env in make_alphazero
        # --- Model ---
        "conv_width": 128,
        "conv_depth": 10,
        "katago_preset": "b10c128nbt",
        "katago_activation": "mish",
        "katago_use_rvgl": True,
        "use_wdl": True,
        # --- MCTS (1sh) ---
        "mcts_num_simulations": 12,
        "mcts_variant": "1sh",
        "mcts_max_m": 8,
        "mcts_num_root_considered": 8,
        "mcts_num_survivors": 4,
        "mcts_num_k_actions": 128,
        "mcts_use_gumbel": True,
        "mcts_gumbel_scale": 1.0,
        "mcts_epsilon": 1e-8,
        "mcts_rescale_values": False,
        "mcts_value_scale": 1.0,
        "mcts_use_mixed_value": True,
        "mcts_maxvisit_init": 50,
        "mcts_bnk_rehydrate_fields": False,
        # bnk: store compressed (K,) policy targets instead of full (4672,)
        "exp_bnk_action_weights": True,
        "exp_use_root_temperature": True,  # from KataGo
        "exp_root_temperature": 1.5,
        # --- Training ---
        "num_iters": 459_000 * 20,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "weight_decay_kernels_only": True,
        "use_bf16": False,
        "lr_warmup_steps": buffer_warmup_steps,
        "train_batch_size": train_bs,
        "cycle_n_selfplay": 20,
        "cycle_n_train": 20,
        # --- Self-play & Buffers ---
        "selfplay_batch_size": selfplay_bs,
        "selfplay_buffer_add_batch_size": selfplay_bs,
        "selfplay_buffer_sample_batch_size": train_bs,
        "selfplay_buffer_min_len": selfplay_buffer_len,
        "selfplay_buffer_max_len": selfplay_buffer_len,
        "selfplay_buffer_consume_size": train_bs,
        "replay_buffer_total_size": REPLAY_BUFFER_TOTAL_SIZE,
        "replay_buffer_add_batch_size": train_bs,
        "replay_buffer_sample_batch_size": train_bs,
        "replay_buffer_min_len": 1,
        "replay_buffer_max_len": replay_buffer_len,
        "replay_buffer_warmup_steps": buffer_warmup_steps,
        "diagnostic_period": 20,
        # --- Strength eval (vs random + frozen anchor; no external engine) ---
        # Plays one game per legal opening (chess = 20) from both colors.
        "eval_period": 50,  # run every N cycles
        # Forced opening depth per eval game: enumerate every legal line of this
        # many plies (1 -> 20 chess openings, 2 -> ~400) and play one game each
        # from both colors. Deeper = wider opening coverage = lower-variance,
        # more accurate ladder score, at proportionally more eval games.
        "eval_opening_plies": 2,
        "eval_max_plies": 200,  # cap match length; unfinished games = draw
        # Max Elo credited for beating random (rung 0's anchor). At 0, beating
        # random just anchors the ladder at Elo 0 and all real Elo comes from
        # beating past selves. Without a cap, ~100% vs random pegs ΔElo at the
        # clamp ceiling (≈+1600) and inflates the whole ladder. Set None to
        # disable the cap entirely.
        "eval_vs_random_max_elo": 0.0,
        "ckpt_period": 800,  # save a checkpoint every 800 cycles
        # --- System ---
        "enable_sharding": True,
        "debug_probe_executables": True,
    }


def get_go_config(board_size=5):
    # pgx1 Go defaults komi to a per-size lookup (all X.5 values, 7.5
    # fallback), so games never draw (score margin is always non-integer):
    # the env always resolves to a win for one side. We treat it like hex
    # (env_forbids_draws=True). The action space is board_size**2 board
    # points + 1 pass, and a game can run up to board_size**2 * 2 plies
    # (pgx1's default max_terminal_steps), so the buffers are sized for that.
    board_cfgs = {
        3: dict(
            conv_width=64,
            conv_depth=4,
            num_iters=(1500 * 10),
            learning_rate=1e-3,
            cycle_n_selfplay=10,
            cycle_n_train=10,
            num_root_considered=8,
            num_survivors=4,
        ),
        4: dict(
            conv_width=128,
            conv_depth=6,
            num_iters=(2000 * 20),
            learning_rate=1e-4,
            cycle_n_selfplay=20,
            cycle_n_train=20,
            num_root_considered=16,
            num_survivors=8,
        ),
        5: dict(
            conv_width=256,
            conv_depth=8,
            num_iters=(5000 * 30),
            learning_rate=1e-4,
            cycle_n_selfplay=30,
            cycle_n_train=25,
            num_root_considered=16,
            num_survivors=8,
        ),
        6: dict(
            conv_width=256,
            conv_depth=8,
            num_iters=(10000 * 40),
            learning_rate=1e-4,
            cycle_n_selfplay=40,
            cycle_n_train=40,
            num_root_considered=8,
            num_survivors=4,
        ),
        7: dict(
            conv_width=256,
            conv_depth=16,
            num_iters=(20000 * 50),
            learning_rate=1e-4,
            cycle_n_selfplay=50,
            cycle_n_train=50,
            num_root_considered=8,
            num_survivors=4,
        ),
        8: dict(
            conv_width=256,
            conv_depth=32,
            num_iters=(50000 * 50),
            learning_rate=1e-4,
            cycle_n_selfplay=50,
            cycle_n_train=50,
            num_root_considered=8,
            num_survivors=4,
        ),
        9: dict(
            conv_width=256,
            conv_depth=16,
            num_iters=(200000 * 50),
            learning_rate=1e-4,
            cycle_n_selfplay=50,
            cycle_n_train=50,
            num_root_considered=8,
            num_survivors=4,
        ),
    }
    if board_size not in board_cfgs:
        raise ValueError(f"Unsupported go board size: {board_size}")
    board_cfg = board_cfgs[board_size]

    # board points + pass; a game can last up to size**2 * 2 plies.
    num_actions = board_size * board_size + 1
    game_max_steps = board_size * board_size * 2

    batch_size = 8192
    REPLAY_BUFFER_TOTAL_SIZE = 2_048_000

    selfplay_buffer_len = game_max_steps + 20
    replay_buffer_len = REPLAY_BUFFER_TOTAL_SIZE // batch_size
    buffer_warmup_steps = (selfplay_buffer_len + replay_buffer_len) * 1

    return {
        # --- Game ---
        "env_id": f"go_{board_size}x{board_size}",
        "game_max_steps": game_max_steps,
        "num_exploratory_moves": game_max_steps // 2,
        # komi 7.5 => no draws (one side always wins on score).
        "env_forbids_draws": True,
        "env_allows_draws": False,
        "boardsize": board_size,
        "game_obs_shape": None,
        "game_num_actions": None,  # patched from live env in make_alphazero
        # --- Model ---
        "conv_width": board_cfg["conv_width"],
        "conv_depth": board_cfg["conv_depth"],
        # --- MCTS (1sh) ---
        "mcts_num_simulations": 24,
        "mcts_variant": "1sh",
        "mcts_max_m": board_cfg["num_root_considered"],
        "mcts_num_root_considered": board_cfg["num_root_considered"],
        "mcts_num_survivors": board_cfg["num_survivors"],
        "mcts_num_k_actions": num_actions,
        "mcts_use_gumbel": True,
        "mcts_gumbel_scale": 1.0,
        "mcts_epsilon": 1e-8,
        "mcts_rescale_values": False,
        "mcts_value_scale": 1.0,
        "mcts_use_mixed_value": True,
        "mcts_maxvisit_init": 50,
        "mcts_bnk_rehydrate_fields": False,
        "exp_bnk_action_weights": False,
        "exp_use_root_temperature": True,
        "exp_root_temperature": 1.3,
        # --- Training ---
        "num_iters": board_cfg["num_iters"],
        "learning_rate": board_cfg["learning_rate"],
        "weight_decay": 1e-4,
        "weight_decay_kernels_only": True,
        "use_bf16": False,
        "lr_warmup_steps": buffer_warmup_steps,
        "train_batch_size": batch_size,
        "cycle_n_selfplay": board_cfg["cycle_n_selfplay"],
        "cycle_n_train": board_cfg["cycle_n_train"],
        # --- Self-play & Buffers ---
        "selfplay_batch_size": batch_size,
        "selfplay_buffer_add_batch_size": batch_size,
        "selfplay_buffer_sample_batch_size": batch_size,
        "selfplay_buffer_min_len": selfplay_buffer_len,
        "selfplay_buffer_max_len": selfplay_buffer_len,
        "selfplay_buffer_consume_size": batch_size,
        "replay_buffer_total_size": REPLAY_BUFFER_TOTAL_SIZE,
        "replay_buffer_add_batch_size": batch_size,
        "replay_buffer_sample_batch_size": batch_size,
        "replay_buffer_min_len": 1,
        "replay_buffer_max_len": replay_buffer_len,
        "replay_buffer_warmup_steps": buffer_warmup_steps,
        # --- Diagnostics & strength eval ---
        "diagnostic_period": 50,
        "eval_period": 50,  # run every N cycles
        "eval_max_plies": None,
        "eval_opening_plies": 2,
        "ckpt_period": None,  # save a checkpoint every N cycles (None = only at end)
        # --- System ---
        "enable_sharding": True,
    }


CONFIG_FACTORIES = {
    "chess": get_chess_config,
    "ttt": get_ttt_config,
    "connect4": get_connect4_config,
    "hex4": lambda: get_hex_config(board_size=4),
    "hex5": lambda: get_hex_config(board_size=5),
    "hex6": lambda: get_hex_config(board_size=6),
    "hex7": lambda: get_hex_config(board_size=7),
    "hex8": lambda: get_hex_config(board_size=8),
    "hex9": lambda: get_hex_config(board_size=9),
    "go3": lambda: get_go_config(board_size=3),
    "go4": lambda: get_go_config(board_size=4),
    "go5": lambda: get_go_config(board_size=5),
    "go6": lambda: get_go_config(board_size=6),
    "go7": lambda: get_go_config(board_size=7),
    "go8": lambda: get_go_config(board_size=8),
    "go9": lambda: get_go_config(board_size=9),
}
