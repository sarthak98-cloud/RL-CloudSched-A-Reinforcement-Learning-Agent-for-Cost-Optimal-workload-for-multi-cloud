"""
workload_parser.py
------------------
Parses and serves workload job traces from:
  - Google Cluster Trace 2019 (Borg format, BigQuery CSV export)
  - Alibaba Cluster Trace 2018 (batch_task.csv format)
  - Synthetic workload generator (used when no traces are available)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


@dataclass
class Job:
    """Single workload job unit. Times in hours, resources in absolute units."""
    job_id:       str
    cpu_demand:   float   # vCPUs
    mem_demand:   float   # GiB
    est_runtime:  float   # hours
    deadline:     float   # relative deadline (hours); -1 = best-effort
    priority:     int     # 1=low, 2=medium, 3=critical
    arrival_time: float   # simulation hour of arrival

    # Set at runtime by the environment
    actual_runtime:  float = 0.0
    assigned_option: int   = -1
    start_time:      float = -1.0
    finish_time:     float = -1.0
    preempted_count: int   = 0
    completed:       bool  = False
    violated:        bool  = False


class WorkloadParser:
    """
    Provides a stream of Job objects for simulation episodes.
    Supports Google 2019, Alibaba 2018, and synthetic traces.
    """

    GOOGLE_MACHINE_VCPU  = 96
    GOOGLE_MACHINE_MEM   = 624
    ALIBABA_MACHINE_VCPU = 96
    ALIBABA_MACHINE_MEM  = 128

    def __init__(
        self,
        trace_type: str = "synthetic",
        data_dir: Optional[str] = None,
        intensity: str = "medium",
        seed: int = 42,
    ):
        self.trace_type = trace_type
        self.data_dir   = Path(data_dir) if data_dir else None
        self.intensity  = intensity
        self.rng        = np.random.default_rng(seed)
        self._arrival_rates = {"light": 5, "medium": 20, "heavy": 80}
        self._rate = self._arrival_rates[intensity]
        self._jobs: List[Job] = []

    def load(self, episode_hours: int = 720):
        self._jobs = []
        if self.trace_type == "google" and self.data_dir:
            self._jobs = self._load_google(episode_hours)
        elif self.trace_type == "alibaba" and self.data_dir:
            self._jobs = self._load_alibaba(episode_hours)
        else:
            self._jobs = self._generate_synthetic(episode_hours)
        self._jobs.sort(key=lambda j: j.arrival_time)

    def jobs_arriving_at(self, hour: float, window: float = 1.0) -> List[Job]:
        return [j for j in self._jobs if hour <= j.arrival_time < hour + window]

    def all_jobs(self) -> List[Job]:
        return list(self._jobs)

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.load()

    # ── Google Cluster Trace 2019 ──────────────────────────────────────

    def _load_google(self, episode_hours: int) -> List[Job]:
        fpath = self.data_dir / "google_jobs.csv"
        if not fpath.exists():
            print(f"[WorkloadParser] Google trace not found. Using synthetic.")
            return self._generate_synthetic(episode_hours)

        df = pd.read_csv(fpath)
        t_min = df["submit_time"].min()
        df["arrival_hours"] = (df["submit_time"] - t_min) / 1e6 / 3600.0
        df = df[df["arrival_hours"] < episode_hours].copy()

        jobs = []
        for _, row in df.iterrows():
            cpu = max(0.25, row.get("requested_cpu", 0.01) * self.GOOGLE_MACHINE_VCPU)
            mem = max(0.5,  row.get("requested_memory", 0.01) * self.GOOGLE_MACHINE_MEM)
            dur = max(0.1,  row.get("duration", 3600) / 3600.0)
            prio_raw = int(row.get("priority", 0))
            priority = 3 if prio_raw >= 9 else (2 if prio_raw >= 5 else 1)
            deadline = dur * 2.0 if priority == 3 else (dur * 5.0 if priority == 2 else -1.0)
            jobs.append(Job(
                job_id=str(row.get("job_id", len(jobs))),
                cpu_demand=round(cpu, 2), mem_demand=round(mem, 2),
                est_runtime=round(dur, 3), deadline=deadline,
                priority=priority, arrival_time=round(float(row["arrival_hours"]), 4),
            ))
        return jobs

    # ── Alibaba Cluster Trace 2018 ─────────────────────────────────────

    def _load_alibaba(self, episode_hours: int) -> List[Job]:
        fpath = self.data_dir / "alibaba_batch_task.csv"
        if not fpath.exists():
            print(f"[WorkloadParser] Alibaba trace not found. Using synthetic.")
            return self._generate_synthetic(episode_hours)

        df = pd.read_csv(fpath)
        t_min = df["start_time"].min()
        df["arrival_hours"] = (df["start_time"] - t_min) / 3600.0
        df = df[
            (df["arrival_hours"] < episode_hours) &
            (df["status"].isin(["Terminated", "Succeeded", "Failed"]))
        ].copy()

        jobs, seen = [], set()
        for _, row in df.iterrows():
            jid = str(row.get("job_name", len(jobs)))
            if jid in seen:
                continue
            seen.add(jid)
            cpu = max(0.25, row.get("plan_cpu", 1.0))
            mem = max(0.5,  row.get("plan_mem", 1.0) * self.ALIBABA_MACHINE_MEM)
            dur = max(0.1,  (row["end_time"] - row["start_time"]) / 3600.0)
            jobs.append(Job(
                job_id=jid, cpu_demand=round(cpu, 2), mem_demand=round(mem, 2),
                est_runtime=round(dur, 3), deadline=dur * 3.0,
                priority=2, arrival_time=round(float(row["arrival_hours"]), 4),
            ))
        return jobs

    # ── Synthetic Generator ────────────────────────────────────────────

    def _generate_synthetic(self, episode_hours: int) -> List[Job]:
        """
        Non-homogeneous Poisson arrivals with diurnal rate variation.
        Resource demands drawn from log-normal distributions calibrated
        to match Google/Alibaba trace statistics.
        """
        jobs, t, idx = [], 0.0, 0
        while t < episode_hours:
            hour_of_day  = t % 24
            rate_mult    = 1.0 + 0.8 * np.sin(2 * np.pi * hour_of_day / 24)
            inter_arrival = self.rng.exponential(1.0 / (self._rate * rate_mult))
            t += inter_arrival
            if t >= episode_hours:
                break

            cpu     = float(np.clip(self.rng.lognormal(0.7, 1.0), 0.25, 64.0))
            mem     = float(np.clip(self.rng.lognormal(1.5, 1.2), 0.5, 256.0))
            runtime = float(np.clip(self.rng.lognormal(0.5, 1.5), 0.1, 72.0))

            p = self.rng.random()
            if p < 0.10:
                priority, deadline = 3, runtime * 2.0
            elif p < 0.80:
                priority, deadline = 2, runtime * 4.0
            else:
                priority, deadline = 1, -1.0

            jobs.append(Job(
                job_id=f"syn_{idx:06d}",
                cpu_demand=round(cpu, 2), mem_demand=round(mem, 2),
                est_runtime=round(runtime, 3), deadline=deadline,
                priority=priority, arrival_time=round(t, 4),
            ))
            idx += 1
        return jobs
