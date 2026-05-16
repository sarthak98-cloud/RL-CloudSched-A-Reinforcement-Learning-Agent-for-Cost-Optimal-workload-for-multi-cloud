"""
config.py
---------
Central configuration for RL-CloudSched experiments.
All values correspond to Table III in the paper (Section IV-I).
"""

# ── MDP Parameters ────────────────────────────────────────────────────
GAMMA                = 0.99    # Discount factor
REWARD_W_COST        = 0.50    # Cost weight
REWARD_W_DEADLINE    = 0.30    # Deadline weight
REWARD_W_PREEMPT     = 0.15    # Preemption weight
REWARD_W_UTIL        = 0.05    # Utilization weight
DEADLINE_CLIP        = 2.0     # alpha_max
CHECKPOINT_HOURS     = 0.05    # ~3 min

# ── PPO Hyperparameters ──────────────────────────────────────────────
PPO_CLIP_EPSILON     = 0.20
PPO_VALUE_COEFF      = 0.50
PPO_ENTROPY_COEFF    = 0.01
PPO_LEARNING_RATE    = 3e-4
PPO_N_STEPS          = 2048
PPO_BATCH_SIZE       = 64
PPO_N_EPOCHS         = 10
PPO_GAE_LAMBDA       = 0.95
PPO_MAX_GRAD_NORM    = 0.5
PPO_TOTAL_TIMESTEPS  = 2_000_000

# ── Network Architecture ─────────────────────────────────────────────
NET_HIDDEN_SIZES     = [256, 128, 64]   # FC layer sizes
NET_ACTIVATION       = "ReLU"

# ── Environment Defaults ─────────────────────────────────────────────
DEFAULT_EPISODE_HOURS = 720     # 30-day episode
DEFAULT_BUDGET        = 5000.0  # USD
DEFAULT_INTENSITY     = "medium"

# ── Experimental Scenarios ───────────────────────────────────────────
SCENARIOS = {
    "light_relaxed_stable": {
        "intensity": "light", "budget": 10000.0,
        "episode_hours": 720,
    },
    "medium_default": {
        "intensity": "medium", "budget": 5000.0,
        "episode_hours": 720,
    },
    "heavy_tight_volatile": {
        "intensity": "heavy", "budget": 3000.0,
        "episode_hours": 720,
    },
}
