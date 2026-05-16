"""
multi_cloud_env.py
------------------
OpenAI Gymnasium-compatible simulation environment for multi-cloud
spot market workload scheduling.

Implements the full MDP defined in Section IV of the paper:
  - State:  ~50-D vector (workload + market + system features)
  - Action: discrete index into (cloud, instance, region) catalog
  - Reward: weighted multi-objective (cost, deadline, preempt, util)

Usage
-----
    from rl_cloudsched import MultiCloudSpotEnv

    env = MultiCloudSpotEnv(intensity="medium", budget=5000.0)
    obs, info = env.reset()

    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Tuple, Any

from rl_cloudsched.utils.instance_catalog import InstanceCatalog
from rl_cloudsched.utils.normalization import FeatureNormalizer, cyclical_encode
from rl_cloudsched.data.spot_price_loader import SpotPriceLoader
from rl_cloudsched.data.workload_parser import WorkloadParser, Job
from rl_cloudsched.environment.preemption import PreemptionEngine
from rl_cloudsched.environment.reward import RewardCalculator


class MultiCloudSpotEnv(gym.Env):
    """
    Custom Gym environment simulating multi-cloud spot market scheduling.

    Observation space: Box(low=0, high=1, shape=(obs_dim,))
    Action space: Discrete(N)  where N = number of resource options

    Parameters
    ----------
    intensity       : workload intensity ("light" / "medium" / "heavy")
    budget          : total budget in USD for the episode
    episode_hours   : simulated duration (hours); default 720 (30 days)
    trace_type      : "synthetic" / "google" / "alibaba"
    trace_dir       : path to trace CSV files (None = synthetic)
    price_dir       : path to spot price CSVs (None = synthetic)
    reward_weights  : dict with keys w_cost, w_deadline, w_preempt, w_util
    seed            : random seed
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    # Number of top market options included in the observation
    # (feasibility-filtered per step)
    TOP_K_OPTIONS = 8

    def __init__(
        self,
        intensity:      str   = "medium",
        budget:         float = 5000.0,
        episode_hours:  int   = 720,
        trace_type:     str   = "synthetic",
        trace_dir:      Optional[str] = None,
        price_dir:      Optional[str] = None,
        reward_weights: Optional[Dict[str, float]] = None,
        seed:           int   = 42,
        render_mode:    Optional[str] = None,
    ):
        super().__init__()
        self.render_mode   = render_mode
        self.budget        = budget
        self.episode_hours = episode_hours
        self.seed_value    = seed

        # ── Sub-components ────────────────────────────────────────────
        self.catalog  = InstanceCatalog()
        self.n_options = len(self.catalog)

        self.price_loader = SpotPriceLoader(
            self.catalog, data_dir=price_dir,
            episode_length_hours=episode_hours, seed=seed,
        )
        self.workload = WorkloadParser(
            trace_type=trace_type, data_dir=trace_dir,
            intensity=intensity, seed=seed,
        )
        self.preemption = PreemptionEngine(seed=seed)

        rw = reward_weights or {}
        self.reward_calc = RewardCalculator(
            w_cost=rw.get("w_cost", 0.50),
            w_deadline=rw.get("w_deadline", 0.30),
            w_preempt=rw.get("w_preempt", 0.15),
            w_util=rw.get("w_util", 0.05),
        )

        # ── Observation & action spaces ───────────────────────────────
        # Obs: 5 (job) + 6*TOP_K (market) + 2*3+3 (system) = 5+48+9 = 62
        self.obs_dim = 5 + 6 * self.TOP_K_OPTIONS + 2 * 3 + 3
        self.observation_space = spaces.Box(
            low=0.0, high=1.5, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_options)

        # ── Episode state ─────────────────────────────────────────────
        self._step_count  = 0
        self._sim_time    = 0.0   # current simulation hour
        self._budget_used = 0.0
        self._job_queue   = []    # List[Job]
        self._current_job: Optional[Job] = None
        self._total_cost  = 0.0
        self._total_jobs_completed  = 0
        self._total_jobs_violated   = 0
        self._total_preemptions     = 0
        self._all_jobs_served       = 0
        self._action_mask = np.ones(self.n_options, dtype=np.int8)

        # Metrics for logging
        self._episode_rewards = []
        self._cost_history    = []

    # ==================================================================
    # Gym API: reset
    # ==================================================================

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment for a new episode."""
        super().reset(seed=seed)

        ep_seed = seed if seed is not None else self.seed_value
        self.price_loader.reset_episode(seed=ep_seed)
        self.price_loader.load()
        self.workload.reset(seed=ep_seed)
        self.workload.load(episode_hours=self.episode_hours)
        self.preemption.clear()

        self._step_count  = 0
        self._sim_time    = 0.0
        self._budget_used = 0.0
        self._total_cost  = 0.0
        self._total_jobs_completed = 0
        self._total_jobs_violated  = 0
        self._total_preemptions    = 0
        self._all_jobs_served      = 0
        self._episode_rewards      = []
        self._cost_history         = []

        # Fill initial job queue with arrivals in the first hour
        self._job_queue = self.workload.jobs_arriving_at(0.0, window=1.0)
        self._advance_to_next_job()

        obs = self._build_observation()
        info = self._build_info()
        return obs, info

    # ==================================================================
    # Gym API: step
    # ==================================================================

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one scheduling decision:
          1. Assign current job to the chosen resource option
          2. Compute immediate reward
          3. Advance simulation time to next decision epoch
          4. Process completions and preemptions
          5. Return new observation
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # ── 1. Apply action masking ───────────────────────────────────
        if self._action_mask[action] == 0:
            # Invalid action: heavy penalty, no scheduling
            obs = self._build_observation()
            return obs, -5.0, False, False, self._build_info()

        # ── 2. Get current job and chosen option ──────────────────────
        job = self._current_job
        opt = self.catalog.get(action)
        prices = self.price_loader.prices_at(int(self._sim_time))
        spot_price = prices.get(action, opt.on_demand_price)

        # ── 3. Compute reward ─────────────────────────────────────────
        abs_deadline = (
            job.arrival_time + job.deadline if job.deadline > 0 else 1e9
        )
        slack = abs_deadline - self._sim_time - job.est_runtime

        reward_components = self.reward_calc.compute(
            spot_price=spot_price,
            on_demand_price=opt.on_demand_price,
            est_runtime=job.est_runtime,
            deadline_slack=slack,
            priority=job.priority,
            eviction_rate=opt.base_eviction_rate,
            job_cpu=job.cpu_demand,
            job_mem=job.mem_demand,
            instance_cpu=opt.vcpu,
            instance_mem=opt.memory_gib,
        )
        reward = reward_components["total"]

        # ── 4. Schedule the job ───────────────────────────────────────
        job.assigned_option = action
        job.start_time = self._sim_time
        self._budget_used += spot_price * job.est_runtime
        self._total_cost  += spot_price * job.est_runtime

        self.preemption.add_job(
            job, option_id=action,
            current_time=self._sim_time,
            abs_deadline=abs_deadline,
        )
        self._all_jobs_served += 1

        # ── 5. Advance simulation ─────────────────────────────────────
        self._step_count += 1
        self._sim_time   += 1.0    # advance 1 hour

        # Process completions and preemptions for this time step
        eviction_rates = {
            o.option_id: o.base_eviction_rate
            for o in self.catalog.all_options()
        }
        completed, preempted = self.preemption.step(
            dt=1.0, eviction_rates=eviction_rates,
            current_time=self._sim_time,
        )

        for rj in completed:
            self._total_jobs_completed += 1
            # Check deadline violation
            if rj.deadline < 1e8 and self._sim_time > rj.deadline:
                self._total_jobs_violated += 1

        for rj in preempted:
            self._total_preemptions += 1
            # Re-queue preempted jobs
            re_job = Job(
                job_id=rj.job_id,
                cpu_demand=rj.cpu_demand,
                mem_demand=rj.mem_demand,
                est_runtime=rj.remaining_hrs,
                deadline=rj.deadline - self._sim_time if rj.deadline < 1e8 else -1.0,
                priority=rj.priority,
                arrival_time=self._sim_time,
                preempted_count=rj.preempted_count,
            )
            self._job_queue.insert(0, re_job)   # priority re-queue

        # Add new arrivals for this hour
        new_arrivals = self.workload.jobs_arriving_at(
            self._sim_time, window=1.0
        )
        self._job_queue.extend(new_arrivals)

        # Advance to next job
        self._advance_to_next_job()

        # ── 6. Check termination ──────────────────────────────────────
        terminated = False
        truncated  = False

        # Budget exhausted
        if self._budget_used >= self.budget:
            terminated = True

        # Episode time limit
        if self._sim_time >= self.episode_hours:
            truncated = True

        # No more jobs and nothing running
        if (
            self._current_job is None
            and len(self._job_queue) == 0
            and self.preemption.n_running == 0
        ):
            terminated = True

        self._episode_rewards.append(reward)
        self._cost_history.append(self._total_cost)

        obs  = self._build_observation()
        info = self._build_info()
        info["reward_components"] = reward_components

        return obs, reward, terminated, truncated, info

    # ==================================================================
    # Observation builder
    # ==================================================================

    def _build_observation(self) -> np.ndarray:
        """
        Construct the observation vector:
          [job_features (5)]
          [market_features (6 * TOP_K)]
          [system_features (2*K + 3)]
        """
        obs = []

        # ── Job features (5D) ─────────────────────────────────────────
        if self._current_job is not None:
            job = self._current_job
            abs_deadline = (
                job.arrival_time + job.deadline if job.deadline > 0 else 1e9
            )
            slack = abs_deadline - self._sim_time - job.est_runtime

            obs.extend([
                np.clip(job.cpu_demand / 64.0, 0, 1),
                np.clip(job.mem_demand / 256.0, 0, 1),
                np.clip(job.est_runtime / 72.0, 0, 1),
                np.clip((slack + 72.0) / 144.0, 0, 1),  # shift to [0,1]
                job.priority / 3.0,
            ])
        else:
            obs.extend([0.0] * 5)

        # ── Market features (6 * TOP_K_OPTIONS) ───────────────────────
        step_int = int(self._sim_time)
        full_market = self.price_loader.market_features(step_int)
        # Reshape to (n_options, 6) and pick TOP_K by lowest price ratio
        market_2d = full_market.reshape(self.n_options, 6)

        # Feasibility mask
        self._update_action_mask()
        feasible_ids = np.where(self._action_mask == 1)[0]

        if len(feasible_ids) > 0:
            # Sort feasible options by price ratio (column 0)
            price_ratios = market_2d[feasible_ids, 0]
            sorted_idx = feasible_ids[np.argsort(price_ratios)]
            top_k = sorted_idx[: self.TOP_K_OPTIONS]
        else:
            top_k = np.arange(min(self.TOP_K_OPTIONS, self.n_options))

        # Pad if fewer than TOP_K
        market_obs = np.zeros(6 * self.TOP_K_OPTIONS, dtype=np.float32)
        for i, oid in enumerate(top_k):
            if i >= self.TOP_K_OPTIONS:
                break
            market_obs[i * 6 : (i + 1) * 6] = market_2d[oid]
        obs.extend(market_obs.tolist())

        # ── System features (2*3 + 3 = 9D) ───────────────────────────
        clouds = ["aws", "azure", "gcp"]

        # Running jobs per cloud (normalized)
        for c in clouds:
            count = self.preemption.running_count_by_cloud(c, self.catalog)
            obs.append(np.clip(count / 100.0, 0, 1))

        # Utilization per cloud
        util_map = self.preemption.utilization(self.catalog)
        for c in clouds:
            obs.append(np.clip(util_map.get(c, 0.0), 0, 1))

        # Queue length (normalized), budget remaining, time encoding
        obs.append(np.clip(len(self._job_queue) / 500.0, 0, 1))
        budget_rem = max(0, self.budget - self._budget_used) / (self.budget + 1e-8)
        obs.append(np.clip(budget_rem, 0, 1))

        # Cyclical hour-of-day encoding (single sin feature)
        hour_sin, _ = cyclical_encode(self._sim_time % 24, 24.0)
        obs.append((hour_sin + 1.0) / 2.0)   # map to [0, 1]

        obs_arr = np.array(obs, dtype=np.float32)
        # Ensure correct dimension
        if len(obs_arr) < self.obs_dim:
            obs_arr = np.pad(obs_arr, (0, self.obs_dim - len(obs_arr)))
        return obs_arr[: self.obs_dim]

    # ==================================================================
    # Action masking
    # ==================================================================

    def _update_action_mask(self):
        """Set action mask based on current job feasibility and budget."""
        self._action_mask = np.zeros(self.n_options, dtype=np.int8)

        if self._current_job is None:
            self._action_mask[:] = 1
            return

        job = self._current_job
        prices = self.price_loader.prices_at(int(self._sim_time))
        budget_rem = self.budget - self._budget_used

        for opt in self.catalog.all_options():
            oid = opt.option_id
            if opt.vcpu < job.cpu_demand:
                continue
            if opt.memory_gib < job.mem_demand:
                continue
            est_cost = prices.get(oid, opt.on_demand_price) * job.est_runtime
            if est_cost > budget_rem:
                continue
            self._action_mask[oid] = 1

        # If nothing is feasible, allow everything (agent must choose least-bad)
        if self._action_mask.sum() == 0:
            self._action_mask[:] = 1

    def action_masks(self) -> np.ndarray:
        """Return current action mask (for MaskablePPO compatibility)."""
        self._update_action_mask()
        return self._action_mask.astype(bool)

    # ==================================================================
    # Job queue management
    # ==================================================================

    def _advance_to_next_job(self):
        """Pop the next job from the queue, or set to None if empty."""
        if len(self._job_queue) > 0:
            # Sort by priority (descending) then arrival time (ascending)
            self._job_queue.sort(
                key=lambda j: (-j.priority, j.arrival_time)
            )
            self._current_job = self._job_queue.pop(0)
        else:
            self._current_job = None

    # ==================================================================
    # Info dict
    # ==================================================================

    def _build_info(self) -> Dict[str, Any]:
        """Build the info dictionary returned at each step."""
        n_served = max(self._all_jobs_served, 1)
        return {
            "sim_time":            self._sim_time,
            "step":                self._step_count,
            "budget_used":         round(self._budget_used, 2),
            "budget_remaining":    round(self.budget - self._budget_used, 2),
            "total_cost":          round(self._total_cost, 2),
            "jobs_completed":      self._total_jobs_completed,
            "jobs_violated":       self._total_jobs_violated,
            "violation_rate":      round(self._total_jobs_violated / n_served, 4),
            "preemptions":         self._total_preemptions,
            "preemption_rate":     round(self._total_preemptions / n_served, 4),
            "jobs_in_queue":       len(self._job_queue),
            "jobs_running":        self.preemption.n_running,
            "total_jobs_served":   self._all_jobs_served,
            "action_mask":         self._action_mask.copy(),
        }

    # ==================================================================
    # Render
    # ==================================================================

    def render(self):
        """Print a human-readable summary of the current state."""
        if self.render_mode is None:
            return

        info = self._build_info()
        job_str = "None"
        if self._current_job is not None:
            j = self._current_job
            job_str = (
                f"{j.job_id} (cpu={j.cpu_demand}, mem={j.mem_demand}, "
                f"runtime={j.est_runtime}h, prio={j.priority})"
            )

        output = (
            f"\n{'='*60}\n"
            f"  Step: {info['step']:>5}  |  Sim Time: {info['sim_time']:.1f}h\n"
            f"  Budget: ${info['budget_used']:.2f} / ${self.budget:.2f}\n"
            f"  Queue: {info['jobs_in_queue']}  |  Running: {info['jobs_running']}\n"
            f"  Completed: {info['jobs_completed']}  |  Violations: {info['jobs_violated']}\n"
            f"  Preemptions: {info['preemptions']}\n"
            f"  Current Job: {job_str}\n"
            f"  Feasible Actions: {int(self._action_mask.sum())}/{self.n_options}\n"
            f"{'='*60}"
        )

        if self.render_mode == "human":
            print(output)
        return output
