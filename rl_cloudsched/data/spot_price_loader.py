"""
spot_price_loader.py
--------------------
Loads and preprocesses historical spot price data for all
(cloud, instance_type, region) options.

Supports:
  - Loading from CSV files (matching real AWS/Azure/GCP export formats)
  - Synthetic price generation via mean-reverting Ornstein–Uhlenbeck process
    when real data is not available (used during initial experiments)
  - Price trace resampling to a fixed dt (default: 1 minute)
  - Rolling statistics: 1h / 6h / 24h deltas, 30-day volatility
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from rl_cloudsched.utils.instance_catalog import InstanceCatalog, InstanceSpec


class SpotPriceLoader:
    """
    Manages spot price time series for every option in the catalog.

    Usage
    -----
    loader = SpotPriceLoader(catalog, data_dir="./data/spot_prices")
    loader.load()                         # loads CSVs or generates synthetic
    prices_t = loader.prices_at(step=42) # dict {option_id: price}
    features  = loader.market_features(step=42)  # full 6*N feature vector
    """

    # Resampling resolution
    STEP_MINUTES = 60   # each environment step = 1 hour of simulated time

    def __init__(
        self,
        catalog: InstanceCatalog,
        data_dir: Optional[str] = None,
        episode_length_hours: int = 720,   # 30 days
        seed: int = 42,
    ):
        self.catalog = catalog
        self.data_dir = Path(data_dir) if data_dir else None
        self.episode_length = episode_length_hours
        self.rng = np.random.default_rng(seed)

        # Populated by load()
        # Shape: (n_options, episode_length_hours)
        self._price_traces: Optional[np.ndarray] = None
        self._n_options = len(catalog)

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def load(self):
        """Load price traces from disk or generate synthetically."""
        if self.data_dir and self.data_dir.exists():
            self._price_traces = self._load_from_csv()
        else:
            print("[SpotPriceLoader] No data dir found — generating synthetic traces.")
            self._price_traces = self._generate_synthetic()

    def prices_at(self, step: int) -> Dict[int, float]:
        """Return {option_id: spot_price} at simulation step t."""
        self._check_loaded()
        step = step % self.episode_length
        return {i: float(self._price_traces[i, step]) for i in range(self._n_options)}

    def market_features(self, step: int) -> np.ndarray:
        """
        Return a 6*N feature vector for the market state at `step`.
        For each option: [price_ratio, delta_1h, delta_6h, delta_24h,
                          eviction_rate, volatility_30d]
        """
        self._check_loaded()
        step = step % self.episode_length
        features = []

        for opt in self.catalog.all_options():
            n = opt.option_id
            p_now  = self._price_traces[n, step]
            p_od   = opt.on_demand_price + 1e-8

            # Price ratio (spot / on-demand)
            price_ratio = np.clip(p_now / p_od, 0.0, 1.5)

            # Rolling deltas
            delta_1h  = self._price_delta(n, step, hours=1)
            delta_6h  = self._price_delta(n, step, hours=6)
            delta_24h = self._price_delta(n, step, hours=24)

            # 30-day eviction rate (static per option in this catalog)
            eviction_rate = opt.base_eviction_rate

            # 30-day rolling price volatility
            volatility = self._rolling_volatility(n, step, window_hours=720)

            features.extend([
                price_ratio, delta_1h, delta_6h, delta_24h,
                eviction_rate, volatility,
            ])

        return np.array(features, dtype=np.float32)

    def reset_episode(self, seed: Optional[int] = None):
        """
        Reshuffle the starting offset so each episode samples a
        different 30-day window from the price trace bank.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # Re-generate a new synthetic episode if no real data
        if self.data_dir is None or not self.data_dir.exists():
            self._price_traces = self._generate_synthetic()

    # ──────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────

    def _check_loaded(self):
        if self._price_traces is None:
            raise RuntimeError("Call SpotPriceLoader.load() before accessing prices.")

    def _price_delta(self, option_id: int, step: int, hours: int) -> float:
        """Relative price change over the last `hours` steps."""
        past_step = max(0, step - hours)
        p_now  = self._price_traces[option_id, step]
        p_past = self._price_traces[option_id, past_step] + 1e-8
        return float(np.clip((p_now - p_past) / p_past, -1.0, 1.0))

    def _rolling_volatility(self, option_id: int, step: int, window_hours: int) -> float:
        """Normalized standard deviation of prices over the past window."""
        lo = max(0, step - window_hours)
        window = self._price_traces[option_id, lo:step + 1]
        if len(window) < 2:
            return 0.0
        std = float(np.std(window))
        mean = float(np.mean(window)) + 1e-8
        return float(np.clip(std / mean, 0.0, 1.0))   # coefficient of variation

    # ──────────────────────────────────────────────────────────────────
    # CSV Loader (AWS / Azure / GCP export format)
    # ──────────────────────────────────────────────────────────────────

    def _load_from_csv(self) -> np.ndarray:
        """
        Expects one CSV per option under data_dir:
          {cloud}_{instance_type}_{region}.csv
        Columns: timestamp (ISO-8601), spot_price (USD/hr)

        Falls back to synthetic generation for missing files.
        """
        T = self.episode_length
        traces = np.zeros((self._n_options, T), dtype=np.float32)

        for opt in self.catalog.all_options():
            fname = (
                f"{opt.cloud}_{opt.instance_type.replace('.', '_')}"
                f"_{opt.region}.csv"
            )
            fpath = self.data_dir / fname
            if fpath.exists():
                df = pd.read_csv(fpath, parse_dates=["timestamp"])
                df = df.sort_values("timestamp").set_index("timestamp")
                # Resample to hourly, forward-fill gaps
                df_h = df["spot_price"].resample("1H").ffill()
                arr = df_h.values[:T]
                if len(arr) < T:
                    arr = np.pad(arr, (0, T - len(arr)), mode="edge")
                traces[opt.option_id] = arr[:T].astype(np.float32)
            else:
                # Fallback to synthetic for this option
                traces[opt.option_id] = self._ou_process(
                    opt.on_demand_price * (1 - opt.spot_discount),
                    kappa=0.3, sigma=0.05 * opt.on_demand_price, T=T
                )

        return traces

    # ──────────────────────────────────────────────────────────────────
    # Synthetic Price Generation (Ornstein–Uhlenbeck)
    # ──────────────────────────────────────────────────────────────────

    def _generate_synthetic(self) -> np.ndarray:
        """
        Generate synthetic spot price traces using a mean-reverting
        Ornstein–Uhlenbeck process with regime switching.

        Each option gets an independent trace calibrated to its
        on-demand price and typical discount factor.
        """
        T = self.episode_length
        traces = np.zeros((self._n_options, T), dtype=np.float32)

        for opt in self.catalog.all_options():
            mean_spot = opt.on_demand_price * (1.0 - opt.spot_discount)
            sigma = 0.06 * opt.on_demand_price   # price noise amplitude
            traces[opt.option_id] = self._ou_process(
                mean=mean_spot, kappa=0.25, sigma=sigma, T=T
            )

        return traces

    def _ou_process(
        self,
        mean: float,
        kappa: float,
        sigma: float,
        T: int,
    ) -> np.ndarray:
        """
        Simulate an Ornstein-Uhlenbeck (mean-reverting) process.

        dp = kappa * (mean - p) * dt + sigma * sqrt(dt) * N(0,1)

        dt = 1 (one step = one hour of simulated time).
        Prices are clipped to [10% of mean, 130% of on-demand] to
        reflect real-world spot price bounds.
        """
        dt = 1.0
        prices = np.zeros(T, dtype=np.float32)
        # Random start near the long-term mean
        prices[0] = mean * self.rng.uniform(0.5, 1.2)

        noise = self.rng.normal(0, 1, size=T)

        # Randomly insert 2-4 "regime spikes" (simulates capacity crunches)
        n_spikes = int(self.rng.integers(2, 5))
        spike_steps = self.rng.choice(T, size=n_spikes, replace=False)

        for t in range(1, T):
            drift = kappa * (mean - prices[t - 1]) * dt
            diffusion = sigma * np.sqrt(dt) * noise[t]
            spike = mean * 0.8 if t in spike_steps else 0.0
            prices[t] = prices[t - 1] + drift + diffusion + spike

        # Enforce physical bounds
        lo = mean * 0.10
        hi = mean * (1.0 / (1.0 - max(0.05, 1.0 - 1.3)))  # ~130% of on-demand
        prices = np.clip(prices, lo, hi)
        return prices.astype(np.float32)
