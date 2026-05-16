"""
preemption.py
-------------
Preemption simulation engine using Poisson hazard rates derived
from historical eviction probabilities per (cloud, instance, region).

At each environment step, running jobs are checked for preemption.
Preempted jobs are returned to the queue with remaining runtime updated.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, field


@dataclass
class RunningJob:
    """Tracks a job currently executing on a cloud instance."""
    job_id:         str
    option_id:      int     # (cloud, instance, region) index
    scheduled_at:   float   # simulation time when job was scheduled
    remaining_hrs:  float   # hours of work remaining
    cpu_demand:     float
    mem_demand:     float
    priority:       int
    deadline:       float   # absolute deadline (arrival + relative deadline)
    preempted_count: int = 0


class PreemptionEngine:
    """
    Simulates preemption events for all currently running jobs.

    Each step, for each running job on option o_n:
      Pr(preemption in dt) = 1 - exp(-lambda_n * dt)
    where lambda_n = -ln(1 - eviction_rate_n)

    Preempted jobs incur a checkpoint overhead and are re-queued.
    """

    CHECKPOINT_OVERHEAD_HRS = 0.05   # ~3 minutes checkpoint time

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self._running: Dict[str, RunningJob] = {}   # job_id -> RunningJob

    # ──────────────────────────────────────────────────────────────────
    # Job lifecycle
    # ──────────────────────────────────────────────────────────────────

    def add_job(self, job, option_id: int, current_time: float, abs_deadline: float):
        """Register a job as running on option_id."""
        from rl_cloudsched.data.workload_parser import Job
        self._running[job.job_id] = RunningJob(
            job_id=job.job_id,
            option_id=option_id,
            scheduled_at=current_time,
            remaining_hrs=job.est_runtime,
            cpu_demand=job.cpu_demand,
            mem_demand=job.mem_demand,
            priority=job.priority,
            deadline=abs_deadline,
            preempted_count=job.preempted_count,
        )

    def remove_job(self, job_id: str):
        """Remove a job that completed normally."""
        self._running.pop(job_id, None)

    def running_jobs(self) -> List[RunningJob]:
        return list(self._running.values())

    def running_count_by_cloud(self, cloud: str, catalog) -> int:
        count = 0
        for rj in self._running.values():
            if catalog.get(rj.option_id).cloud == cloud:
                count += 1
        return count

    # ──────────────────────────────────────────────────────────────────
    # Step simulation
    # ──────────────────────────────────────────────────────────────────

    def step(
        self,
        dt: float,
        eviction_rates: Dict[int, float],
        current_time: float,
    ) -> Tuple[List[RunningJob], List[RunningJob]]:
        """
        Advance simulation by dt hours.

        Returns:
            completed   - jobs that finished naturally this step
            preempted   - jobs that were evicted this step
        """
        completed  = []
        preempted  = []
        still_running = {}

        for job_id, rj in self._running.items():
            # Progress the job
            rj.remaining_hrs -= dt

            if rj.remaining_hrs <= 0:
                # Job finished naturally
                rj.remaining_hrs = 0.0
                completed.append(rj)
                continue

            # Check for preemption using Poisson hazard rate
            eviction_rate = eviction_rates.get(rj.option_id, 0.05)
            hazard = -np.log(1.0 - min(eviction_rate, 0.9999))
            p_preempt = 1.0 - np.exp(-hazard * dt)

            if self.rng.random() < p_preempt:
                rj.preempted_count += 1
                # Add checkpoint overhead to remaining time
                rj.remaining_hrs += self.CHECKPOINT_OVERHEAD_HRS
                preempted.append(rj)
            else:
                still_running[job_id] = rj

        self._running = still_running
        return completed, preempted

    def clear(self):
        """Reset for new episode."""
        self._running.clear()

    @property
    def n_running(self) -> int:
        return len(self._running)

    def utilization(self, catalog) -> Dict[str, float]:
        """
        Compute per-cloud average CPU utilization as a fraction of
        total deployed capacity across running jobs.
        """
        cloud_cpu_used = {}
        cloud_cpu_total = {}
        for rj in self._running.values():
            opt = catalog.get(rj.option_id)
            c = opt.cloud
            cloud_cpu_used[c]  = cloud_cpu_used.get(c, 0.0)  + rj.cpu_demand
            cloud_cpu_total[c] = cloud_cpu_total.get(c, 0.0) + opt.vcpu
        result = {}
        for c in cloud_cpu_used:
            denom = cloud_cpu_total.get(c, 1.0) + 1e-8
            result[c] = cloud_cpu_used[c] / denom
        return result
