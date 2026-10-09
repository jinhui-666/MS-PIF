from __future__ import annotations

from datetime import timezone
import importlib
from pathlib import Path
import sys
from types import ModuleType

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DGL_ROOT = PROJECT_ROOT / "backbones" / "DGL"

APPEND_INFORMATION = [
    True,
    True,
    True,
    False,
    True,
    True,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
]

MODEL_PARAMETERS = {
    "mode": "test",
    "embedding_dim": 128,
    "sqrt_embedding_dim": 128**0.5,
    "decoder_layer_num": 3,
    "qkv_dim": 16,
    "head_num": 8,
    "ff_hidden_dim": 512,
    "append_information": APPEND_INFORMATION,
}


def _install_optional_import_shims() -> None:
    try:
        importlib.import_module("matplotlib.pyplot")
    except ModuleNotFoundError:
        pyplot = ModuleType("matplotlib.pyplot")
        matplotlib = ModuleType("matplotlib")
        matplotlib.pyplot = pyplot
        sys.modules["matplotlib"] = matplotlib
        sys.modules["matplotlib.pyplot"] = pyplot
    try:
        importlib.import_module("pytz")
    except ModuleNotFoundError:
        pytz = ModuleType("pytz")
        pytz.timezone = lambda _name: timezone.utc
        sys.modules["pytz"] = pytz


def load_model(checkpoint: Path, device: torch.device):
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    for path in (PROJECT_ROOT, PROJECT_ROOT / "backbones", DGL_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    _install_optional_import_shims()
    from DGL.CVRP.CVRPModel import CVRPModel

    model = CVRPModel(**MODEL_PARAMETERS).to(device)
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def choose_candidates(model, candidates):
    """Apply the official encoder/decoder weights to one equal-width group."""
    encoded = model.encoder(candidates.model_input)
    k = candidates.direct_indices.shape[1]
    direct_nodes = encoded[:, :k]
    direct_last = model.decoder.embedding_last_node_not_via_depot(encoded[:, k])
    depot_nodes = encoded[:, k + 1 : -1]
    depot_last = model.decoder.embedding_last_node_via_depot(encoded[:, -1])
    values = torch.cat(
        (
            direct_last[:, None, :],
            direct_nodes,
            depot_last[:, None, :],
            depot_nodes,
        ),
        dim=1,
    )
    for layer in model.decoder.layers:
        values = layer(values)
    logits = model.decoder.Linear_final(values).squeeze(-1)
    logits[:, 0] = float("-inf")
    logits[:, k + 1] = float("-inf")
    logits[:, 1 : k + 1] += candidates.direct_ninf_mask
    actionable = torch.cat((logits[:, 1 : k + 1], logits[:, k + 2 :]), dim=1)
    choices = actionable.argmax(dim=1)
    via_depot = choices >= k
    local = torch.where(via_depot, choices - k, choices)
    row_axis = torch.arange(len(local), device=local.device)
    nodes = torch.where(
        via_depot,
        candidates.depot_indices[row_axis, local],
        candidates.direct_indices[row_axis, local],
    )
    return nodes, via_depot
