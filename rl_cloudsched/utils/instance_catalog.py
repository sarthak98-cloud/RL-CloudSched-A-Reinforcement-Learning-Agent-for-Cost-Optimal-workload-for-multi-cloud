"""
instance_catalog.py
-------------------
Defines the full catalog of cloud instances across AWS, Azure, and GCP
with on-demand prices, CPU, memory, and default eviction rates.
Used by the environment and data loader.
"""

from dataclasses import dataclass, field
from typing import List, Dict


# ─────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────

@dataclass
class InstanceSpec:
    """Specification for a single (cloud, instance_type, region) option."""
    option_id:       int     # Unique index in the action space
    cloud:           str     # "aws" | "azure" | "gcp"
    instance_type:   str     # e.g. "m5.xlarge"
    region:          str     # e.g. "us-east-1"
    vcpu:            float   # Number of vCPUs
    memory_gib:      float   # Memory in GiB
    on_demand_price: float   # On-demand hourly price in USD
    # 30-day historical eviction rate (fraction, 0-1)
    base_eviction_rate: float = 0.05
    # Spot-to-on-demand discount fraction used for synthetic price generation
    spot_discount:   float = 0.70


class InstanceCatalog:
    """
    Catalog of all (cloud, instance_type, region) resource options.

    In a real deployment these values come from cloud provider APIs.
    Here we hardcode a representative ~75-option catalog covering the
    three major clouds and their most common compute families.
    """

    def __init__(self):
        self._options: List[InstanceSpec] = []
        self._build_catalog()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def all_options(self) -> List[InstanceSpec]:
        return self._options

    def get(self, option_id: int) -> InstanceSpec:
        return self._options[option_id]

    def __len__(self):
        return len(self._options)

    def filter_feasible(
        self,
        min_vcpu: float,
        min_mem: float,
        max_cost_per_hour: float,
        current_prices: Dict[int, float],
    ) -> List[int]:
        """
        Return option IDs whose capacity meets (min_vcpu, min_mem)
        and whose current spot price is within max_cost_per_hour.
        """
        result = []
        for opt in self._options:
            if opt.vcpu < min_vcpu:
                continue
            if opt.memory_gib < min_mem:
                continue
            price = current_prices.get(opt.option_id, opt.on_demand_price)
            if price > max_cost_per_hour:
                continue
            result.append(opt.option_id)
        return result

    # ------------------------------------------------------------------
    # Catalog Construction (representative dataset)
    # ------------------------------------------------------------------

    def _build_catalog(self):
        idx = 0

        # ── AWS ──────────────────────────────────────────────────────
        aws_specs = [
            # (instance_type, region,        vcpu, mem,  od_price, eviction)
            ("m5.large",    "us-east-1",     2,    8,    0.096,  0.05),
            ("m5.xlarge",   "us-east-1",     4,    16,   0.192,  0.05),
            ("m5.2xlarge",  "us-east-1",     8,    32,   0.384,  0.04),
            ("m5.4xlarge",  "us-east-1",     16,   64,   0.768,  0.04),
            ("c5.xlarge",   "us-east-1",     4,    8,    0.170,  0.06),
            ("c5.2xlarge",  "us-east-1",     8,    16,   0.340,  0.06),
            ("c5.4xlarge",  "us-east-1",     16,   32,   0.680,  0.05),
            ("r5.xlarge",   "us-east-1",     4,    32,   0.252,  0.04),
            ("r5.2xlarge",  "us-east-1",     8,    64,   0.504,  0.04),
            ("m5.large",    "us-west-2",     2,    8,    0.096,  0.07),
            ("m5.xlarge",   "us-west-2",     4,    16,   0.192,  0.07),
            ("c5.2xlarge",  "us-west-2",     8,    16,   0.340,  0.08),
            ("m5.2xlarge",  "eu-west-1",     8,    32,   0.416,  0.05),
            ("c5.xlarge",   "eu-west-1",     4,    8,    0.192,  0.07),
            ("r5.xlarge",   "ap-southeast-1",4,    32,   0.296,  0.06),
            ("m5.large",    "ap-southeast-1",2,    8,    0.117,  0.06),
            ("g4dn.xlarge", "us-east-1",     4,    16,   0.526,  0.10),
            ("g4dn.2xlarge","us-east-1",     8,    32,   1.052,  0.10),
            ("p3.2xlarge",  "us-east-1",     8,    61,   3.060,  0.12),
            ("m5.large",    "eu-central-1",  2,    8,    0.107,  0.05),
            ("c5.4xlarge",  "us-west-2",     16,   32,   0.680,  0.07),
            ("r5.4xlarge",  "us-east-1",     16,   128,  1.008,  0.03),
        ]
        for (itype, region, vcpu, mem, od, ev) in aws_specs:
            self._options.append(InstanceSpec(
                option_id=idx, cloud="aws",
                instance_type=itype, region=region,
                vcpu=vcpu, memory_gib=mem,
                on_demand_price=od, base_eviction_rate=ev,
                spot_discount=0.70,
            ))
            idx += 1

        # ── Azure ─────────────────────────────────────────────────────
        azure_specs = [
            # (instance_type,  region,         vcpu, mem,  od_price, eviction)
            ("Standard_D2s_v3",  "eastus",     2,    8,    0.096,  0.06),
            ("Standard_D4s_v3",  "eastus",     4,    16,   0.192,  0.06),
            ("Standard_D8s_v3",  "eastus",     8,    32,   0.384,  0.05),
            ("Standard_D16s_v3", "eastus",     16,   64,   0.768,  0.04),
            ("Standard_F4s_v2",  "eastus",     4,    8,    0.169,  0.07),
            ("Standard_F8s_v2",  "eastus",     8,    16,   0.338,  0.07),
            ("Standard_E4s_v3",  "eastus",     4,    32,   0.252,  0.05),
            ("Standard_E8s_v3",  "eastus",     8,    64,   0.504,  0.04),
            ("Standard_D4s_v3",  "westeurope", 4,    16,   0.211,  0.06),
            ("Standard_D8s_v3",  "westeurope", 8,    32,   0.422,  0.05),
            ("Standard_F4s_v2",  "westeurope", 4,    8,    0.188,  0.08),
            ("Standard_D2s_v3",  "southeastasia",2,  8,    0.105,  0.07),
            ("Standard_D4s_v3",  "southeastasia",4,  16,   0.210,  0.07),
            ("Standard_NC6s_v3", "eastus",     6,    112,  3.060,  0.12),
            ("Standard_D32s_v3", "eastus",     32,   128,  1.536,  0.03),
            ("Standard_E16s_v3", "eastus",     16,   128,  1.008,  0.04),
            ("Standard_F16s_v2", "westus2",    16,   32,   0.676,  0.08),
        ]
        for (itype, region, vcpu, mem, od, ev) in azure_specs:
            self._options.append(InstanceSpec(
                option_id=idx, cloud="azure",
                instance_type=itype, region=region,
                vcpu=vcpu, memory_gib=mem,
                on_demand_price=od, base_eviction_rate=ev,
                spot_discount=0.68,
            ))
            idx += 1

        # ── GCP ───────────────────────────────────────────────────────
        gcp_specs = [
            # (instance_type,       region,             vcpu, mem,  od_price, eviction)
            ("n1-standard-2",   "us-central1",      2,    7.5,  0.095,  0.08),
            ("n1-standard-4",   "us-central1",      4,    15,   0.190,  0.08),
            ("n1-standard-8",   "us-central1",      8,    30,   0.380,  0.07),
            ("n1-standard-16",  "us-central1",      16,   60,   0.760,  0.06),
            ("n2-standard-4",   "us-central1",      4,    16,   0.194,  0.07),
            ("n2-standard-8",   "us-central1",      8,    32,   0.388,  0.07),
            ("c2-standard-4",   "us-central1",      4,    16,   0.209,  0.09),
            ("c2-standard-8",   "us-central1",      8,    32,   0.418,  0.09),
            ("n1-highmem-4",    "us-central1",      4,    26,   0.237,  0.06),
            ("n1-highmem-8",    "us-central1",      8,    52,   0.474,  0.05),
            ("n1-standard-4",   "europe-west4",     4,    15,   0.209,  0.08),
            ("n1-standard-8",   "europe-west4",     8,    30,   0.418,  0.08),
            ("n2-standard-4",   "us-east1",         4,    16,   0.194,  0.09),
            ("n1-standard-2",   "asia-southeast1",  2,    7.5,  0.111,  0.09),
            ("a2-highgpu-1g",   "us-central1",      12,   85,   3.673,  0.13),
            ("n1-standard-32",  "us-central1",      32,   120,  1.520,  0.05),
            ("n2-highcpu-8",    "us-central1",      8,    8,    0.311,  0.10),
            ("n2-standard-16",  "us-central1",      16,   64,   0.776,  0.06),
        ]
        for (itype, region, vcpu, mem, od, ev) in gcp_specs:
            self._options.append(InstanceSpec(
                option_id=idx, cloud="gcp",
                instance_type=itype, region=region,
                vcpu=vcpu, memory_gib=mem,
                on_demand_price=od, base_eviction_rate=ev,
                spot_discount=0.72,
            ))
            idx += 1
