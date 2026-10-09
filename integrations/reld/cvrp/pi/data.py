from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import torch
import vrplib


@dataclass
class CVRPInstance:
    name: str
    coords: torch.Tensor
    demands: torch.Tensor
    capacity: float
    bks: Optional[float] = None

    @property
    def customer_count(self) -> int:
        return int(self.coords.shape[0]) - 1


@dataclass
class CVRPBatch:
    names: List[str]
    model_coords: torch.Tensor
    original_coords: torch.Tensor
    normalized_demands: torch.Tensor
    node_mask: torch.Tensor
    customer_counts: torch.Tensor
    capacities: torch.Tensor
    bks: Optional[torch.Tensor]

    @property
    def batch_size(self) -> int:
        return int(self.model_coords.shape[0])

    @property
    def max_nodes(self) -> int:
        return int(self.model_coords.shape[1])

    @property
    def max_customers(self) -> int:
        return self.max_nodes - 1


@dataclass
class BucketConfig:
    upper_bound: int
    pomo_size: int
    aug_factor: int


@dataclass
class Bucket:
    config: BucketConfig
    instances: List[CVRPInstance]


def _depot_index(raw_depot) -> int:
    depot = torch.as_tensor(raw_depot).reshape(-1)
    if depot.numel() == 0:
        return 0
    return int(depot[0].item())


def load_instance(path: Path) -> CVRPInstance:
    raw = vrplib.read_instance(path)
    coords = torch.as_tensor(raw["node_coord"], dtype=torch.float32)
    demands = torch.as_tensor(raw["demand"], dtype=torch.float32)
    depot_index = _depot_index(raw.get("depot", [0]))
    if depot_index < 0 or depot_index >= coords.shape[0]:
        raise ValueError(f"{path}: invalid depot index {depot_index}.")
    if depot_index != 0:
        order = [depot_index] + [idx for idx in range(coords.shape[0]) if idx != depot_index]
        coords = coords[order]
        demands = demands[order]
    demands = demands.clone()
    demands[0] = 0.0

    solution_path = path.with_suffix(".sol")
    bks = None
    if solution_path.exists():
        solution = vrplib.read_solution(solution_path)
        if solution.get("cost") is not None:
            bks = float(solution["cost"])
    return CVRPInstance(path.stem, coords, demands, float(raw["capacity"]), bks)


def load_instances(path: Path, limit: Optional[int] = None) -> List[CVRPInstance]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    paths = sorted(path.glob("*.vrp")) if path.is_dir() else [path]
    instances = [load_instance(item) for item in paths]
    instances.sort(key=lambda item: (item.customer_count, item.name))
    return instances[:limit] if limit is not None else instances


def make_buckets(instances: Sequence[CVRPInstance], configs: Sequence[BucketConfig]) -> List[Bucket]:
    if not configs:
        raise ValueError("At least one bucket configuration is required.")
    sorted_configs = sorted(configs, key=lambda config: config.upper_bound)
    bounds = [config.upper_bound for config in sorted_configs]
    if len(bounds) != len(set(bounds)):
        raise ValueError("Bucket upper bounds must be unique.")
    for config in sorted_configs:
        if config.upper_bound <= 0 or config.pomo_size <= 0:
            raise ValueError("Bucket upper bound and POMO size must be positive.")
        if config.aug_factor not in (1, 8):
            raise ValueError("Each bucket aug_factor must be 1 or 8.")
    buckets = [Bucket(config, []) for config in sorted_configs]
    for instance in sorted(instances, key=lambda item: (item.customer_count, item.name)):
        for bucket in buckets:
            if instance.customer_count <= bucket.config.upper_bound:
                bucket.instances.append(instance)
                break
        else:
            raise ValueError(
                f"{instance.name} has {instance.customer_count} customers, exceeding "
                f"the largest bucket {bounds[-1]}."
            )
    return [bucket for bucket in buckets if bucket.instances]


def pad_instances(instances: Sequence[CVRPInstance], device: torch.device) -> CVRPBatch:
    if not instances:
        raise ValueError("Cannot pad an empty instance list.")
    batch_size = len(instances)
    max_nodes = max(instance.coords.shape[0] for instance in instances)
    model_coords = torch.zeros(batch_size, max_nodes, 2, dtype=torch.float32, device=device)
    original_coords = torch.zeros_like(model_coords)
    normalized_demands = torch.zeros(batch_size, max_nodes, dtype=torch.float32, device=device)
    node_mask = torch.ones(batch_size, max_nodes, dtype=torch.bool, device=device)
    customer_counts = torch.empty(batch_size, dtype=torch.long, device=device)
    capacities = torch.empty(batch_size, dtype=torch.float32, device=device)
    has_bks = all(instance.bks is not None for instance in instances)
    bks_values = []

    for row, instance in enumerate(instances):
        count = int(instance.coords.shape[0])
        coords = instance.coords.to(device=device, dtype=torch.float32)
        demands = instance.demands.to(device=device, dtype=torch.float32)
        coord_min = coords.min(dim=0, keepdim=True).values
        coord_span = (coords.max(dim=0, keepdim=True).values - coord_min).clamp_min(1e-12)
        model_coords[row, :count] = (coords - coord_min) / coord_span
        original_coords[row, :count] = coords
        normalized_demands[row, :count] = demands / float(instance.capacity)
        node_mask[row, :count] = False
        customer_counts[row] = instance.customer_count
        capacities[row] = float(instance.capacity)
        if has_bks:
            bks_values.append(float(instance.bks))

    bks = torch.tensor(bks_values, dtype=torch.float32, device=device) if has_bks else None
    return CVRPBatch(
        names=[instance.name for instance in instances],
        model_coords=model_coords,
        original_coords=original_coords,
        normalized_demands=normalized_demands,
        node_mask=node_mask,
        customer_counts=customer_counts,
        capacities=capacities,
        bks=bks,
    )
