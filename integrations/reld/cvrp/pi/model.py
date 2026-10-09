import math
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones.reld.CVRP.CVRPModel import CVRPModel


DEFAULT_MODEL_PARAMS = {
    "embedding_dim": 128,
    "encoder_layer_num": 6,
    "head_num": 8,
    "qkv_dim": 16,
    "forcing_first_step": False,
    "logit_clipping": 50,
    "ff_hidden_dim": 512,
    "eval_type": "greedy",
}


def stable_argmax(values: torch.Tensor, relative_tolerance: float) -> torch.Tensor:
    """Resolve numerically near-equal greedy choices by the lowest node index."""
    maximum = values.max(dim=-1, keepdim=True).values
    tolerance = float(relative_tolerance) * maximum.abs()
    eligible = values >= maximum - tolerance
    indices = torch.arange(values.shape[-1], device=values.device)
    indices = indices.view(*([1] * (values.ndim - 1)), -1).expand_as(values)
    sentinel = torch.full_like(indices, values.shape[-1])
    return torch.where(eligible, indices, sentinel).min(dim=-1).values


def stable_topk(
    values: torch.Tensor,
    k: int,
    relative_tolerance: float,
) -> torch.Tensor:
    remaining = values.clone()
    selected = []
    for _ in range(int(k)):
        index = stable_argmax(remaining, relative_tolerance)
        selected.append(index)
        remaining.scatter_(1, index.unsqueeze(1), float("-inf"))
    return torch.stack(selected, dim=1)


def reshape_by_heads(qkv: torch.Tensor, head_num: int) -> torch.Tensor:
    batch_size, node_num, _ = qkv.shape
    return qkv.reshape(batch_size, node_num, head_num, -1).transpose(1, 2)


def masked_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_mask: torch.Tensor,
) -> torch.Tensor:
    score = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(q.shape[-1])
    score = score.masked_fill(key_mask[:, None, None, :], torch.finfo(score.dtype).min)
    weights = torch.softmax(score, dim=-1).masked_fill(key_mask[:, None, None, :], 0.0)
    out = torch.matmul(weights, v)
    return out.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)


def masked_attention_rank3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ninf_mask: torch.Tensor,
) -> torch.Tensor:
    score = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(q.shape[-1])
    score = score + ninf_mask[:, None, :, :]
    weights = torch.softmax(score, dim=-1)
    out = torch.matmul(weights, v)
    return out.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)


class MaskedEncoderLayer(nn.Module):
    def __init__(self, source_layer: nn.Module, head_num: int):
        super().__init__()
        self.head_num = head_num
        self.Wq = source_layer.Wq
        self.Wk = source_layer.Wk
        self.Wv = source_layer.Wv
        self.multi_head_combine = source_layer.multi_head_combine
        self.feed_forward = source_layer.feed_forward

    def forward(self, tokens: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        q = reshape_by_heads(self.Wq(tokens), self.head_num)
        k = reshape_by_heads(self.Wk(tokens), self.head_num)
        v = reshape_by_heads(self.Wv(tokens), self.head_num)
        attended = masked_attention(q, k, v, node_mask)
        tokens = tokens + self.multi_head_combine(attended)
        tokens = tokens + self.feed_forward(tokens)
        return tokens.masked_fill(node_mask.unsqueeze(-1), 0.0)


class MaskedReLDModel(nn.Module):
    def __init__(
        self,
        source: CVRPModel,
        *,
        greedy_tie_relative_tolerance: float,
        **model_params,
    ):
        super().__init__()
        self.model_params = model_params
        self.greedy_tie_relative_tolerance = float(greedy_tie_relative_tolerance)
        if self.greedy_tie_relative_tolerance < 0:
            raise ValueError("greedy_tie_relative_tolerance must be non-negative.")
        self.forcing_first_step = bool(model_params["forcing_first_step"])
        self.embedding_depot = source.encoder.embedding_depot
        self.embedding_node = source.encoder.embedding_node
        self.encoder_layers = nn.ModuleList(
            MaskedEncoderLayer(layer, model_params["head_num"]) for layer in source.encoder.layers
        )
        decoder = source.decoder
        self.Wq_last = decoder.Wq_last
        self.Wk = decoder.Wk
        self.Wv = decoder.Wv
        self.multi_head_combine = decoder.multi_head_combine
        self.capacity_mapping = decoder.capacity_mapping
        self.feed_forward = decoder.feed_forward
        self.k = None
        self.v = None
        self.single_head_key = None
        self.node_mask = None
        self.encoded_nodes = None
        self.pomo_valid_mask = None

    def pre_forward(
        self,
        depot_xy: torch.Tensor,
        node_xy: torch.Tensor,
        node_demand: torch.Tensor,
        node_mask: torch.Tensor,
        pomo_valid_mask: torch.Tensor,
    ) -> None:
        customer_features = torch.cat((node_xy, node_demand.unsqueeze(-1)), dim=2)
        tokens = torch.cat(
            (self.embedding_depot(depot_xy), self.embedding_node(customer_features)), dim=1
        )
        tokens = tokens.masked_fill(node_mask.unsqueeze(-1), 0.0)
        for layer in self.encoder_layers:
            tokens = layer(tokens, node_mask)
        self.encoded_nodes = tokens
        self.node_mask = node_mask
        self.pomo_valid_mask = pomo_valid_mask
        self.k = reshape_by_heads(self.Wk(tokens), self.model_params["head_num"])
        self.v = reshape_by_heads(self.Wv(tokens), self.model_params["head_num"])
        self.single_head_key = tokens.transpose(1, 2)

    def _probs(
        self,
        current_node: torch.Tensor,
        load: torch.Tensor,
        cur_dist: torch.Tensor,
        ninf_mask: torch.Tensor,
    ) -> torch.Tensor:
        gather_index = current_node.unsqueeze(-1).expand(-1, -1, self.encoded_nodes.shape[-1])
        encoded_last = self.encoded_nodes.gather(1, gather_index)
        query_input = torch.cat((encoded_last, load.unsqueeze(-1)), dim=2)
        q = reshape_by_heads(self.Wq_last(query_input), self.model_params["head_num"])
        out = masked_attention_rank3(q, self.k, self.v, ninf_mask)
        refined = self.multi_head_combine(out) + encoded_last + self.capacity_mapping(load.unsqueeze(-1))
        refined = self.feed_forward(refined) + refined
        score = torch.matmul(refined, self.single_head_key)
        score = score / math.sqrt(self.model_params["embedding_dim"])
        score = score - torch.log(cur_dist)
        score = self.model_params["logit_clipping"] * torch.tanh(score)
        score = score + ninf_mask
        return F.softmax(score, dim=2).masked_fill(self.node_mask[:, None, :], 0.0)

    @torch.no_grad()
    def one_step_rollout(self, state, cur_dist: Optional[torch.Tensor]) -> torch.Tensor:
        batch_size, pomo_size = self.pomo_valid_mask.shape
        device = self.pomo_valid_mask.device
        if state.selected_count == 0:
            return torch.zeros(batch_size, pomo_size, dtype=torch.long, device=device)
        if state.selected_count == 1 and self.forcing_first_step:
            raise NotImplementedError("PI currently supports ReLD greedy diverse-first-step inference only.")

        probs = self._probs(state.current_node, state.load, cur_dist, state.ninf_mask)
        if state.selected_count == 1:
            top_index = stable_topk(
                probs[:, 0, :],
                pomo_size,
                self.greedy_tie_relative_tolerance,
            )
            fallback = top_index[:, :1].expand_as(top_index)
            return torch.where(self.pomo_valid_mask, top_index, fallback)
        return stable_argmax(probs, self.greedy_tie_relative_tolerance)


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    model_params: Optional[Dict] = None,
    greedy_tie_relative_tolerance: float = 0.0,
) -> MaskedReLDModel:
    params = dict(DEFAULT_MODEL_PARAMS)
    if model_params:
        params.update(model_params)
    source = CVRPModel(**params).to(device)
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    source.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    source.eval()
    model = MaskedReLDModel(
        source,
        greedy_tie_relative_tolerance=greedy_tie_relative_tolerance,
        **params,
    ).to(device)
    model.eval()
    return model
