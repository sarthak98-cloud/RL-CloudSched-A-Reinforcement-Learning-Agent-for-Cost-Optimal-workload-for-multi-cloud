"""
test_env.py
-----------
Smoke test: creates the environment, runs 100 random steps,
and prints a summary. Validates that all components integrate correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rl_cloudsched import MultiCloudSpotEnv


def main():
    print("=" * 60)
    print("  RL-CloudSched — Environment Smoke Test")
    print("=" * 60)

    env = MultiCloudSpotEnv(
        intensity="medium",
        budget=5000.0,
        episode_hours=200,   # Short episode for testing
        trace_type="synthetic",
        seed=42,
        render_mode="human",
    )

    print(f"\nCatalog size (action space): {env.n_options}")
    print(f"Observation dim:            {env.obs_dim}")
    print(f"Observation space:          {env.observation_space}")
    print(f"Action space:               {env.action_space}")

    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    env.render()

    total_reward = 0.0
    n_steps = 100

    for step in range(n_steps):
        # Sample from feasible actions
        mask = env.action_masks()
        feasible = mask.nonzero()[0]
        if len(feasible) > 0:
            action = int(feasible[env.np_random.integers(len(feasible))])
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 25 == 0:
            env.render()

        if terminated or truncated:
            print(f"\nEpisode ended at step {step + 1}")
            break

    print(f"\n{'=' * 60}")
    print(f"  RESULTS after {min(step + 1, n_steps)} steps")
    print(f"{'=' * 60}")
    print(f"  Total reward:     {total_reward:.4f}")
    print(f"  Total cost:       ${info['total_cost']:.2f}")
    print(f"  Budget used:      ${info['budget_used']:.2f}")
    print(f"  Jobs completed:   {info['jobs_completed']}")
    print(f"  Jobs violated:    {info['jobs_violated']}")
    print(f"  Violation rate:   {info['violation_rate']:.2%}")
    print(f"  Preemptions:      {info['preemptions']}")
    print(f"  Preemption rate:  {info['preemption_rate']:.2%}")
    print(f"  Jobs served:      {info['total_jobs_served']}")
    print(f"{'=' * 60}")
    print("\n  All checks passed.")


if __name__ == "__main__":
    main()
