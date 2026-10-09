"""Single padding-aware rollout path shared by the adapter and standalone tool."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import CVRPBatch
from .env import PaddedReLDEnv
from .model import MaskedReLDModel


@dataclass(frozen=True)
class PaddedInferenceResult:
    costs: torch.Tensor
    no_aug_costs: torch.Tensor
    sequences: torch.Tensor
    effective_pomo: int
    environment: PaddedReLDEnv


@torch.no_grad()
def solve_padded_batch(
    model: MaskedReLDModel,
    batch: CVRPBatch,
    *,
    pomo_size: int,
    aug_factor: int,
) -> PaddedInferenceResult:
    """Run exactly one mixed-size ReLD batch."""
    effective_pomo = min(int(pomo_size), batch.max_customers)
    env = PaddedReLDEnv(
        batch,
        pomo_size=effective_pomo,
        aug_factor=aug_factor,
    )
    model.pre_forward(
        depot_xy=env.depot_xy,
        node_xy=env.node_xy,
        node_demand=env.node_demand,
        node_mask=env.node_mask,
        pomo_valid_mask=env.pomo_valid_mask,
    )
    state = env.reset()
    done = False
    while not done:
        selected = model.one_step_rollout(state, env.get_cur_feature())
        state, _, done = env.step(selected)
    costs, no_aug_costs, sequences = env.best_results()
    return PaddedInferenceResult(
        costs=costs,
        no_aug_costs=no_aug_costs,
        sequences=sequences,
        effective_pomo=effective_pomo,
        environment=env,
    )
