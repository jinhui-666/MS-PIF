from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import torch

from .data import PaddedBatch
from .env import DGLPIEnvironment
from .model import choose_candidates


@torch.no_grad()
def solve(model, batch: PaddedBatch, *, starts: Sequence[int]) -> DGLPIEnvironment:
    environment = DGLPIEnvironment(batch)
    environment.start(torch.as_tensor(starts, dtype=torch.long, device=batch.data.device))
    while bool(environment.active_mask.any()):
        active_rows = torch.nonzero(environment.active_mask, as_tuple=False).flatten()
        groups: dict[int, list[int]] = defaultdict(list)
        for row in active_rows.detach().cpu().tolist():
            remaining = int(environment.remaining_counts[row])
            groups[min(50, remaining)].append(row)
        for k in sorted(groups, reverse=True):
            rows = torch.as_tensor(groups[k], dtype=torch.long, device=batch.data.device)
            candidates = environment.candidate_batch(rows, k)
            nodes, via_depot = choose_candidates(model, candidates)
            environment.select(rows, nodes, via_depot)
    return environment
