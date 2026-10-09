"""One greedy, padding-aware rollout with batch-local inference caches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .data import PaddedTSPBatch
from .env import PaddedTSPEnvironment
from .model import DGLTSPFeatures, DGLTSPModel, load_model


@dataclass(frozen=True)
class PaddedTSPInference:
    tours: tuple[torch.Tensor, ...]


@torch.no_grad()
def solve_padded_batch(
    model: DGLTSPModel,
    batch: PaddedTSPBatch,
    starts: Sequence[int] | torch.Tensor,
    knn_size: int = 30,
) -> PaddedTSPInference:
    if not isinstance(knn_size, int) or isinstance(knn_size, bool) or knn_size < 1:
        raise ValueError("knn_size must be a positive integer.")
    environment = PaddedTSPEnvironment(batch)
    state = environment.reset(torch.as_tensor(starts, device=batch.normalized_coords.device))
    features = DGLTSPFeatures(batch, state)
    done = not bool(state.active_mask.any().item())
    while not done:
        selected = model(batch, state, features.build(state), knn_size)
        features.advance(selected, state.active_mask)
        state, done = environment.step(selected)
    return PaddedTSPInference(tours=environment.tours())
