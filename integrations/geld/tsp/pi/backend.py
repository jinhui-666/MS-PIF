"""One padding-aware greedy rollout shared by the adapter and tools."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import TSPBatch
from .env import PaddedTSPEnv
from .model import MaskedGELDModel


@dataclass(frozen=True)
class PaddedTSPInferenceResult:
    tours: tuple[torch.Tensor, ...]
    environment: PaddedTSPEnv


@torch.no_grad()
def solve_padded_batch(
    model: MaskedGELDModel,
    batch: TSPBatch,
) -> PaddedTSPInferenceResult:
    env = PaddedTSPEnv(batch)
    state = env.reset()
    model.pre_forward(batch)
    done = not bool(torch.any(state.active_mask).item())
    while not done:
        selected = model.greedy_step(state)
        state, done = env.step(selected)
    return PaddedTSPInferenceResult(
        tours=env.best_tours(),
        environment=env,
    )
