from dataclasses import dataclass
from typing import List, Tuple

import torch

from .data import CVRPBatch


@dataclass
class StepState:
    selected_count: int
    load: torch.Tensor
    current_node: torch.Tensor | None
    ninf_mask: torch.Tensor
    finished: torch.Tensor


def augment_xy_data(coords: torch.Tensor, aug_factor: int) -> torch.Tensor:
    if aug_factor == 1:
        return coords
    if aug_factor != 8:
        raise ValueError("ReLD supports aug_factor 1 or 8.")
    x = coords[:, :, [0]]
    y = coords[:, :, [1]]
    variants = (
        torch.cat((x, y), dim=2),
        torch.cat((1 - x, y), dim=2),
        torch.cat((x, 1 - y), dim=2),
        torch.cat((1 - x, 1 - y), dim=2),
        torch.cat((y, x), dim=2),
        torch.cat((1 - y, x), dim=2),
        torch.cat((y, 1 - x), dim=2),
        torch.cat((1 - y, 1 - x), dim=2),
    )
    return torch.cat(variants, dim=0)


class PaddedReLDEnv:
    def __init__(self, batch: CVRPBatch, pomo_size: int, aug_factor: int):
        self.batch = batch
        self.base_batch_size = batch.batch_size
        self.aug_factor = int(aug_factor)
        self.problem_size = batch.max_customers
        self.pomo_size = min(int(pomo_size), self.problem_size)
        if self.pomo_size <= 0:
            raise ValueError("pomo_size must be positive.")

        self.model_coords = augment_xy_data(batch.model_coords, self.aug_factor)
        self.original_coords = batch.original_coords.repeat(self.aug_factor, 1, 1)
        self.demands = batch.normalized_demands.repeat(self.aug_factor, 1)
        self.node_mask = batch.node_mask.repeat(self.aug_factor, 1)
        self.customer_counts = batch.customer_counts.repeat(self.aug_factor)
        self.batch_size = int(self.model_coords.shape[0])
        rank = torch.arange(self.pomo_size, device=self.model_coords.device).unsqueeze(0)
        valid_width = self.customer_counts.clamp_max(self.pomo_size).unsqueeze(1)
        self.pomo_valid_mask = rank < valid_width
        self.reset()

    @property
    def depot_xy(self) -> torch.Tensor:
        return self.model_coords[:, :1, :]

    @property
    def node_xy(self) -> torch.Tensor:
        return self.model_coords[:, 1:, :]

    @property
    def node_demand(self) -> torch.Tensor:
        return self.demands[:, 1:]

    def reset(self) -> StepState:
        device = self.model_coords.device
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.zeros(
            self.batch_size, self.pomo_size, 0, dtype=torch.long, device=device
        )
        self.at_the_depot = torch.ones(
            self.batch_size, self.pomo_size, dtype=torch.bool, device=device
        )
        self.load = torch.ones(self.batch_size, self.pomo_size, device=device)
        padding_mask = self.node_mask[:, None, :].expand(-1, self.pomo_size, -1)
        self.visited_ninf_flag = torch.zeros(
            self.batch_size, self.pomo_size, self.problem_size + 1, device=device
        ).masked_fill(padding_mask, float("-inf"))
        self.ninf_mask = self.visited_ninf_flag.clone()
        self.finished = torch.zeros(
            self.batch_size, self.pomo_size, dtype=torch.bool, device=device
        )
        return self._state()

    def _state(self) -> StepState:
        return StepState(
            selected_count=self.selected_count,
            load=self.load,
            current_node=self.current_node,
            ninf_mask=self.ninf_mask,
            finished=self.finished,
        )

    def get_cur_feature(self) -> torch.Tensor | None:
        if self.current_node is None:
            return None
        gather_index = self.current_node.unsqueeze(-1).expand(-1, -1, 2)
        current_coords = self.model_coords.gather(1, gather_index)
        return (
            current_coords.unsqueeze(2) - self.model_coords.unsqueeze(1)
        ).norm(p=2, dim=-1)

    def step(self, selected: torch.Tensor) -> Tuple[StepState, torch.Tensor | None, bool]:
        self.selected_count += 1
        self.current_node = selected
        self.selected_node_list = torch.cat(
            (self.selected_node_list, selected.unsqueeze(2)), dim=2
        )
        self.at_the_depot = selected == 0

        demand_list = self.demands[:, None, :].expand(-1, self.pomo_size, -1)
        selected_demand = demand_list.gather(2, selected.unsqueeze(2)).squeeze(2)
        self.load = self.load - selected_demand
        self.load[self.at_the_depot] = 1.0

        self.visited_ninf_flag.scatter_(2, self.selected_node_list, float("-inf"))
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0.0
        self.ninf_mask = self.visited_ninf_flag.clone()
        demand_too_large = self.load.unsqueeze(2) + 1e-6 < demand_list
        self.ninf_mask[demand_too_large] = float("-inf")

        newly_finished = (self.visited_ninf_flag == float("-inf")).all(dim=2)
        self.finished = self.finished | newly_finished
        self.ninf_mask[:, :, 0][self.finished] = 0.0
        self.visited_ninf_flag[:, :, 0][self.finished] = 0.0
        done = bool(self.finished.all().item())
        reward = -self.route_costs() if done else None
        return self._state(), reward, done

    def route_costs(self) -> torch.Tensor:
        gather_index = self.selected_node_list.unsqueeze(-1).expand(-1, -1, -1, 2)
        coords = self.original_coords[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        ordered = coords.gather(2, gather_index)
        rolled = ordered.roll(dims=2, shifts=-1)
        segment_lengths = torch.round((ordered - rolled).norm(p=2, dim=3))
        return segment_lengths.sum(dim=2)

    def best_results(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        costs = self.route_costs().masked_fill(~self.pomo_valid_mask, float("inf"))
        costs = costs.reshape(self.aug_factor, self.base_batch_size, self.pomo_size)
        pomo_cost, pomo_index = costs.min(dim=2)
        best_cost, aug_index = pomo_cost.min(dim=0)
        batch_index = torch.arange(self.base_batch_size, device=costs.device)
        best_pomo = pomo_index[aug_index, batch_index]

        solutions = self.selected_node_list.reshape(
            self.aug_factor, self.base_batch_size, self.pomo_size, -1
        )
        best_solutions = solutions[aug_index, batch_index, best_pomo]
        no_aug_cost = costs[0].min(dim=1).values
        return best_cost, no_aug_cost, best_solutions

    def check_legality(self, solutions: torch.Tensor) -> List[dict]:
        solutions = solutions.detach().cpu()
        demands = self.batch.normalized_demands.detach().cpu()
        counts = self.batch.customer_counts.detach().cpu()
        results = []
        for row, name in enumerate(self.batch.names):
            count = int(counts[row].item())
            route = solutions[row].tolist()
            customers = [node for node in route if node != 0]
            reasons = []
            if sorted(customers) != list(range(1, count + 1)):
                reasons.append("customers_not_visited_exactly_once")
            load = 0.0
            max_load = 0.0
            for step, node in enumerate(route):
                if node == 0:
                    load = 0.0
                elif 1 <= node <= count:
                    load += float(demands[row, node].item())
                    max_load = max(max_load, load)
                    if load > 1.0 + 1e-5:
                        reasons.append(f"capacity_exceeded_at_{step}")
                        break
                else:
                    reasons.append(f"invalid_node_{node}")
                    break
            results.append(
                {
                    "name": name,
                    "is_legal": not reasons,
                    "illegal_reason": "OK" if not reasons else ";".join(reasons),
                    "max_load_ratio": max_load,
                }
            )
        return results
