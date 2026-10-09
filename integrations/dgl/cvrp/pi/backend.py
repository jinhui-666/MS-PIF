from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from .data import PaddedBatch
from .model import load_model
from .tester import solve


@dataclass(frozen=True)
class PIInference:
    sequences: torch.Tensor
    via_depot: torch.Tensor


def load_checkpoint(checkpoint: Path, device: torch.device):
    return load_model(checkpoint, device)


def solve_padded_batch(
    model,
    batch: PaddedBatch,
    *,
    starts: Sequence[int],
) -> PIInference:
    environment = solve(model, batch, starts=starts)
    return PIInference(
        sequences=environment.sequence,
        via_depot=environment.via_depot,
    )
