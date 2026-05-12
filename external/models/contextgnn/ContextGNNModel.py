"""
This file is adapted from code originally released in the ContextGNN project
and/or in the Rel-DeepLearning-RecSys repository.

Original sources:
- https://github.com/kumo-ai/ContextGNN
- https://github.com/danielemalitesta/Rel-DeepLearning-RecSys

The original code is distributed under the MIT License. The corresponding
copyright and license notices are preserved in this repository.

Modifications were made for the experiments accompanying this submission.
"""

from typing import Any, Dict, Optional, Type
import torch
from torch import Tensor
from torch_frame.data.stats import StatType
from torch_frame.nn.models import ResNet
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType
from typing_extensions import Self
from .nn.encoder import (
    DEFAULT_STYPE_ENCODER_DICT,
    HeteroEncoder,
    HeteroTemporalEncoder
)
from .nn.models import HeteroGraphSAGE, HeteroNGCF, HeteroLightGCN
from .nn.models.rhsembeddinggnn import RHSEmbeddingGNN
from .utils import RHSEmbeddingMode
import numpy as np
import math

class ContextGNNModel(RHSEmbeddingGNN):
    r"""Implementation of HybridGNN model."""

    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        rhs_emb_mode: RHSEmbeddingMode,
        dst_entity_table: str,
        num_nodes: int,
        num_layers: int,
        channels: int,
        embedding_dim: int,
        dropout: float,
        normalize: bool,
        aggr: str = "sum",
        norm: str = "layer_norm",
        lr: float = 1e-3,
        gnn: str = "GraphSAGE",
        eps: float = 1e-5,
        train_eps: bool = True,
        torch_frame_model_cls: Type[torch.nn.Module] = ResNet,
        torch_frame_model_kwargs: Optional[Dict[str, Any]] = None,
        is_static: Optional[bool] = False,
        num_src_nodes: Optional[int] = None,
        src_entity_table: Optional[str] = None,
        fusion_gate_version: str = "v1",
        gate_temperature: float = 1.0
    ) -> None:
        super().__init__(
            data,
            col_stats_dict,
            rhs_emb_mode,
            dst_entity_table,
            num_nodes,
            embedding_dim,
            num_src_nodes,
            src_entity_table
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
            stype_encoder_cls_kwargs=DEFAULT_STYPE_ENCODER_DICT,
            torch_frame_model_cls=torch_frame_model_cls,
            torch_frame_model_kwargs=torch_frame_model_kwargs
        )

        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[
                node_type for node_type in data.node_types if "time" in data[node_type]
            ],
            channels=channels
        )

        if gnn == "GraphSAGE":
            self.gnn = HeteroGraphSAGE(
                node_types=data.node_types,
                edge_types=data.edge_types,
                channels=channels,
                aggr=aggr,
                num_layers=num_layers
            )
        elif gnn == "NGCF":
            self.gnn = HeteroNGCF(
                node_types=data.node_types,
                edge_types=data.edge_types,
                channels=channels,
                aggr=aggr,
                dropout=dropout,
                num_layers=num_layers
            )
        elif gnn == "LightGCN":
            self.gnn = HeteroLightGCN(
                node_types=data.node_types,
                edge_types=data.edge_types,
                aggr=aggr,
                normalize=normalize,
                num_layers=num_layers
            )
        else:
            raise NotImplementedError
        
        self.head = MLP(
            channels,
            out_channels=1,
            norm=norm,
            num_layers=1
        )

        self.lhs_projector = torch.nn.Linear(channels, embedding_dim)
        self.id_awareness_emb = torch.nn.Embedding(1, channels)

        self.fusion_gate_version = str(fusion_gate_version).lower().strip()

        self._gate_temperature_init = float(gate_temperature)

        if self._gate_temperature_init <= 0:
            raise ValueError("gate_temperature must be > 0")

        allowed_fgv = {"v1", "v2", "v3"}

        if self.fusion_gate_version not in allowed_fgv:
            raise ValueError(
                f"fusion_gate_version='{self.fusion_gate_version}' not valid. "
                f"Admitted: {sorted(allowed_fgv)}"
            )

        if self.fusion_gate_version == "v1":
            self.gate_mlp = torch.nn.Sequential(
                torch.nn.Linear(2 * channels, channels),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.1),
                torch.nn.Linear(channels, 1)
            )

            self.gate_norm = None
            self.norm_u_local = None
            self.norm_u_global = None
            self.norm_v_local = None
            self.log_gate_temperature = None
        else:
            gate_emb_dim = (2 * channels) + embedding_dim

            self.gate_mlp = torch.nn.Sequential(
                torch.nn.Linear(gate_emb_dim + 1, channels // 2),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=0.1),
                torch.nn.Linear(channels // 2, 1)
            )

            self.gate_norm = torch.nn.LayerNorm(gate_emb_dim)
            self.norm_u_local = torch.nn.LayerNorm(channels)
            self.norm_u_global = torch.nn.LayerNorm(embedding_dim)
            self.norm_v_local = torch.nn.LayerNorm(channels)

            if self.fusion_gate_version == "v3":
                self.log_gate_temperature = torch.nn.Parameter(
                    torch.tensor(
                        math.log(self._gate_temperature_init),
                        device=self.device,
                        dtype=torch.float32
                    )
                )
            else:
                self.log_gate_temperature = None

        self.channels = channels
        self.is_static = is_static
        self.reset_parameters()
        
        self.gnn.to(self.device)
        self.lhs_projector.to(self.device)
        self.rhs_embedding.to(self.device)
        self.head.to(self.device)
        self.id_awareness_emb.to(self.device)
        
        self.encoder.to(self.device)
        self.temporal_encoder.to(self.device)
        
        if self.lhs_embedding is not None:
            self.lhs_embedding.to(self.device)
        
        self.gate_mlp.to(self.device)

        if self.gate_norm is not None:
            self.gate_norm.to(self.device)

        if self.norm_u_local is not None:
            self.norm_u_local.to(self.device)

        if self.norm_u_global is not None:
            self.norm_u_global.to(self.device)

        if self.norm_v_local is not None:
            self.norm_v_local.to(self.device)
        
        self._monitor = {}
        self._last_local_assign = None
        self._last_gate_tensor = None
        self._last_gate_assign = None

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def reset_parameters(self) -> None:
        super().reset_parameters()

        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()
        self.id_awareness_emb.reset_parameters()
        self.rhs_embedding.reset_parameters()
        self.lhs_projector.reset_parameters()

        for m in self.gate_mlp.modules():
            if isinstance(m, torch.nn.Linear):
                m.reset_parameters()

        if self.gate_norm is not None:
            self.gate_norm.reset_parameters()

        if self.norm_u_local is not None:
            self.norm_u_local.reset_parameters()

        if self.norm_u_global is not None:
            self.norm_u_global.reset_parameters()

        if self.norm_v_local is not None:
            self.norm_v_local.reset_parameters()

        if self.log_gate_temperature is not None:
            with torch.no_grad():
                self.log_gate_temperature.fill_(math.log(self._gate_temperature_init))

        if self.lhs_embedding is not None:
            self.lhs_embedding.reset_parameters()

    def _get_gate_temperature(self) -> Tensor:
        if self.log_gate_temperature is None:
            raise RuntimeError("Gate temperature requested but fusion_gate_version is not v3.")

        return torch.exp(self.log_gate_temperature).clamp(min=1e-4, max=1e4)

    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType,
        dst_table: NodeType,
        score_mode: str = "full",
        fixed_gate_value: Optional[float] = None
    ) -> Tensor:
        score_mode = str(score_mode).lower().strip()

        if fixed_gate_value is not None:
            fixed_gate_value = float(fixed_gate_value)

            if not (0.0 <= fixed_gate_value <= 1.0):
                raise ValueError("fixed_gate_value must be in [0, 1].")

            if self.training:
                raise ValueError("fixed_gate_value can only be used during inference.")

        seed_time = batch[entity_table].seed_time
        
        x_dict = self.encoder(batch.tf_dict)
        
        if self.lhs_embedding is not None:
            lhs = self.lhs_embedding() 
            x_dict[entity_table] = lhs[batch[entity_table].n_id]

        id_aw = self.id_awareness_emb.weight
        x_dict[entity_table][:seed_time.size(0)] += id_aw

        if not self.is_static:
            rel_time_dict = self.temporal_encoder(
                seed_time,
                batch.time_dict,
                batch.batch_dict
            )

            for node_type, rel_time in rel_time_dict.items():
                x_dict[node_type] = x_dict[node_type] + rel_time

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict
        )
        
        batch_size = seed_time.size(0)
        lhs_embedding = x_dict[entity_table][:batch_size]
        lhs_embedding_projected = self.lhs_projector(lhs_embedding)
        rhs_gnn_embedding = x_dict[dst_table]
        rhs_idgnn_index = batch.n_id_dict[dst_table]
        lhs_idgnn_batch = batch.batch_dict[dst_table]
        
        rhs_embedding = self.rhs_embedding()

        embgnn_logits = torch.matmul(lhs_embedding_projected, rhs_embedding.t())

        embgnn_logits_pre_mean = embgnn_logits.mean()
        embgnn_logits_pre_std = embgnn_logits.std()

        global_selected = embgnn_logits[lhs_idgnn_batch, rhs_idgnn_index]

        idgnn_logits = self.head(rhs_gnn_embedding).flatten()
        idgnn_logits += (lhs_embedding[lhs_idgnn_batch] * rhs_gnn_embedding).sum(dim=-1)

        delta_override = (idgnn_logits - global_selected)
        delta_override_mean = delta_override.mean()
        delta_override_std = delta_override.std()

        num_items = rhs_embedding.size(0)
        num_overrides = rhs_idgnn_index.numel()
        override_ratio = num_overrides / (batch_size * num_items)

        local_per_user = torch.bincount(lhs_idgnn_batch, minlength=batch_size).float()
        local_per_user_mean = local_per_user.mean()
        local_per_user_std = local_per_user.std(unbiased=False)

        if score_mode == "full":
            if fixed_gate_value is not None:
                g_id = torch.full_like(idgnn_logits, fill_value=fixed_gate_value)
            else:
                if self.fusion_gate_version == "v1":
                    pair_feat = torch.cat(
                        [lhs_embedding[lhs_idgnn_batch], rhs_gnn_embedding],
                        dim=-1
                    )

                    gate_logit = self.gate_mlp(pair_feat).flatten()
                    g_id = torch.sigmoid(gate_logit)
                else:
                    u_local = lhs_embedding[lhs_idgnn_batch]
                    u_global = lhs_embedding_projected[lhs_idgnn_batch]
                    v_local = rhs_gnn_embedding

                    u_local_n = self.norm_u_local(u_local)
                    u_global_n = self.norm_u_global(u_global)
                    v_local_n = self.norm_v_local(v_local)

                    emb_feat = torch.cat([u_local_n, u_global_n, v_local_n], dim=-1)
                    emb_feat = self.gate_norm(emb_feat)

                    pair_feat = torch.cat([emb_feat, delta_override.unsqueeze(-1)], dim=-1)

                    gate_logit = self.gate_mlp(pair_feat).flatten()

                    if self.fusion_gate_version == "v3":
                        tau = self._get_gate_temperature()
                        g_id = torch.sigmoid(gate_logit / tau)
                    else:
                        g_id = torch.sigmoid(gate_logit)

            self._last_gate_tensor = g_id

            fused_logits = g_id * idgnn_logits + (1.0 - g_id) * global_selected
            embgnn_logits[lhs_idgnn_batch, rhs_idgnn_index] = fused_logits
        elif score_mode == "global":
            self._last_gate_tensor = None
        elif score_mode == "local":
            local_only = torch.full(
                (batch_size, rhs_embedding.size(0)),
                float("-inf"),
                device=embgnn_logits.device,
                dtype=embgnn_logits.dtype
            )

            local_only[lhs_idgnn_batch, rhs_idgnn_index] = idgnn_logits
            embgnn_logits = local_only

            self._last_gate_tensor = None
        else:
            raise ValueError(f"Unknown score_mode='{score_mode}'. Use one of: full, global, local.")

        if self.training:
            with torch.no_grad():
                lhs_norm = lhs_embedding.norm(p=2, dim=1)
                rhs_gnn_norm = rhs_gnn_embedding.norm(p=2, dim=1)
                rhs_emb_norm = rhs_embedding.norm(p=2, dim=1)

                lhs_disp = lhs_embedding.var(dim=0, unbiased=False).mean()
                rhs_gnn_disp = rhs_gnn_embedding.var(dim=0, unbiased=False).mean()
                rhs_emb_disp = rhs_embedding.var(dim=0, unbiased=False).mean()

                emb_logits_mean = embgnn_logits.mean()
                emb_logits_std = embgnn_logits.std()
                id_logits_mean = idgnn_logits.mean()
                id_logits_std = idgnn_logits.std()

                g = self._last_gate_tensor

                gate_mean = g.mean()
                gate_std = g.std()
                gate_min = g.min()
                gate_max = g.max()

                eps = 1e-8
                gate_entropy = -(g * torch.log(g + eps) + (1.0 - g) * torch.log(1.0 - g + eps)).mean()

                self._monitor = {
                    "lhs_norm_mean": lhs_norm.mean().item(),
                    "lhs_norm_std": lhs_norm.std().item(),
                    "rhs_gnn_norm_mean": rhs_gnn_norm.mean().item(),
                    "rhs_gnn_norm_std": rhs_gnn_norm.std().item(),
                    "rhs_emb_norm_mean": rhs_emb_norm.mean().item(),
                    "rhs_emb_norm_std": rhs_emb_norm.std().item(),
                    "lhs_dispersion": lhs_disp.item(),
                    "rhs_gnn_dispersion": rhs_gnn_disp.item(),
                    "rhs_emb_dispersion": rhs_emb_disp.item(),
                    "emb_logit_mean": emb_logits_mean.item(),
                    "emb_logit_std": emb_logits_std.item(),
                    "id_logit_mean": id_logits_mean.item(),
                    "id_logit_std": id_logits_std.item(),
                    "emb_logit_pre_mean": embgnn_logits_pre_mean.item(),
                    "emb_logit_pre_std": embgnn_logits_pre_std.item(),
                    "override_ratio": float(override_ratio),
                    "num_overrides": int(num_overrides),
                    "num_items": int(num_items),
                    "local_items_per_user_mean": local_per_user_mean.item(),
                    "local_items_per_user_std": local_per_user_std.item(),
                    "delta_override_mean": delta_override_mean.item(),
                    "delta_override_std": delta_override_std.item(),
                    "gate_mean": gate_mean.item(),
                    "gate_std": gate_std.item(),
                    "gate_min": gate_min.item(),
                    "gate_max": gate_max.item(),
                    "gate_entropy_mean": gate_entropy.item()
                }

                if self.fusion_gate_version == "v3":
                    self._monitor["gate_temperature"] = float(self._get_gate_temperature().detach().item())
        
        if not self.training:
            self._last_local_assign = (
                lhs_idgnn_batch.detach().cpu(),
                rhs_idgnn_index.detach().cpu(),
                batch_size
            )

            if score_mode == "full" and self._last_gate_tensor is not None:
                self._last_gate_assign = (
                    lhs_idgnn_batch.detach().cpu(),
                    rhs_idgnn_index.detach().cpu(),
                    self._last_gate_tensor.detach().cpu(),
                    batch_size
                )
            else:
                self._last_gate_assign = None
        else:
            self._last_local_assign = None
            self._last_gate_assign = None

        return embgnn_logits

    def to(self, *args, **kwargs) -> Self:
        return super().to(*args, **kwargs)

    def cpu(self) -> Self:
        return super().cpu()

    def cuda(self, *args, **kwargs) -> Self:
        return super().cuda(*args, **kwargs)

    def get_top_k(self, preds, train_mask, k=100):
        if not torch.is_tensor(preds):
            preds = torch.as_tensor(preds)

        if preds.device != self.device:
            preds = preds.to(self.device, non_blocking=True)

        mask = torch.as_tensor(train_mask, device=preds.device, dtype=torch.bool)
        preds.masked_fill_(~mask, float("-inf"))

        return torch.topk(preds, k=k, dim=1, sorted=True)