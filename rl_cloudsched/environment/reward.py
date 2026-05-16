"""
reward.py
---------
Multi-objective reward function for RL-CloudSched.

R(s, a) = w1 * R_cost + w2 * R_deadline + w3 * R_preempt + w4 * R_util

Each component is normalized to roughly [-1, 0] or [0, 1] so that the
weights directly control relative importance.
"""

import numpy as np
from typing import Dict, Optional


class RewardCalculator:
    """
    Computes the composite reward signal for a scheduling action.

    Parameters
    ----------
    w_cost      : weight for cost minimization          (default 0.50)
    w_deadline  : weight for deadline compliance         (default 0.30)
    w_preempt   : weight for preemption risk avoidance   (default 0.15)
    w_util      : weight for resource utilization bonus  (default 0.05)
    alpha_max   : clipping constant for deadline penalty (default 2.0)
    ckpt_hours  : checkpoint overhead in hours            (default 0.05)
    """

    def __init__(
        self,
        w_cost:     float = 0.50,
        w_deadline: float = 0.30,
        w_preempt:  float = 0.15,
        w_util:     float = 0.05,
        alpha_max:  float = 2.0,
        ckpt_hours: float = 0.05,
    ):
        self.w_cost     = w_cost
        self.w_deadline = w_deadline
        self.w_preempt  = w_preempt
        self.w_util     = w_util
        self.alpha_max  = alpha_max
        self.ckpt_hours = ckpt_hours

    def compute(
        self,
        spot_price:       float,
        on_demand_price:  float,
        est_runtime:      float,
        deadline_slack:   float,
        priority:         int,
        eviction_rate:    float,
        job_cpu:          float,
        job_mem:          float,
        instance_cpu:     float,
        instance_mem:     float,
    ) -> Dict[str, float]:
        """
        Compute the full reward and its decomposition.

        Parameters
        ----------
        spot_price      : current spot price of the chosen option ($/hr)
        on_demand_price : on-demand price of the chosen option ($/hr)
        est_runtime     : estimated runtime of the job (hours)
        deadline_slack  : deadline - now - est_runtime (hours); negative = overdue
        priority        : job priority (1, 2, or 3)
        eviction_rate   : 30-day eviction probability of the chosen option [0, 1]
        job_cpu         : job CPU demand (vCPUs)
        job_mem         : job memory demand (GiB)
        instance_cpu    : instance CPU capacity (vCPUs)
        instance_mem    : instance memory capacity (GiB)

        Returns
        -------
        dict with keys: "total", "cost", "deadline", "preempt", "util"
        """
        r_cost     = self._cost_reward(spot_price, on_demand_price)
        r_deadline = self._deadline_reward(deadline_slack, est_runtime, priority)
        r_preempt  = self._preempt_reward(eviction_rate, est_runtime)
        r_util     = self._util_reward(job_cpu, job_mem, instance_cpu, instance_mem)

        total = (
            self.w_cost     * r_cost
            + self.w_deadline * r_deadline
            + self.w_preempt  * r_preempt
            + self.w_util     * r_util
        )

        return {
            "total":    total,
            "cost":     r_cost,
            "deadline": r_deadline,
            "preempt":  r_preempt,
            "util":     r_util,
        }

    # ──────────────────────────────────────────────────────────────────
    # Component functions
    # ──────────────────────────────────────────────────────────────────

    def _cost_reward(self, spot_price: float, on_demand_price: float) -> float:
        """
        R_cost = -(spot_price / on_demand_price)
        Range: [-1.5, 0] (clipped; ratio > 1 means spot exceeds on-demand)
        Closer to 0 = bigger savings.
        """
        ratio = spot_price / (on_demand_price + 1e-8)
        return -float(np.clip(ratio, 0.0, 1.5))

    def _deadline_reward(
        self, slack: float, est_runtime: float, priority: int
    ) -> float:
        """
        R_deadline = 0                                        if slack >= 0
                   = -priority * min(|slack| / est_runtime, alpha_max)  if slack < 0

        Normalized by est_runtime so a 1-hour job missing by 1 hour is
        penalized equally to a 10-hour job missing by 10 hours.
        """
        if slack >= 0 or est_runtime <= 0:
            return 0.0
        severity = min(abs(slack) / (est_runtime + 1e-8), self.alpha_max)
        return -float(priority * severity)

    def _preempt_reward(self, eviction_rate: float, est_runtime: float) -> float:
        """
        R_preempt = -eviction_rate * (1 + ckpt_overhead / est_runtime)

        Anticipatory penalty: penalizes risky options at schedule-time,
        providing denser reward signal than post-hoc eviction penalties.
        Short jobs get higher relative checkpoint overhead.
        """
        overhead_ratio = self.ckpt_hours / (est_runtime + 1e-8)
        return -float(eviction_rate * (1.0 + overhead_ratio))

    def _util_reward(
        self, job_cpu: float, job_mem: float,
        inst_cpu: float, inst_mem: float
    ) -> float:
        """
        R_util = (job_cpu / inst_cpu) * (job_mem / inst_mem)

        Rewards tight-packing: selecting an instance whose capacity
        closely matches the job's needs. Product ensures both dimensions
        are well-utilized.
        Range: (0, 1]
        """
        cpu_ratio = min(job_cpu / (inst_cpu + 1e-8), 1.0)
        mem_ratio = min(job_mem / (inst_mem + 1e-8), 1.0)
        return float(cpu_ratio * mem_ratio)
