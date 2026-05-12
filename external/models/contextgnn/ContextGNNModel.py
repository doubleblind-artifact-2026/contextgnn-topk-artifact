"""
This file is adapted from code originally released in the ContextGNN project.

Original source:
- https://github.com/kumo-ai/ContextGNN

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
        feature_type: str = "const_original"
    ) -> None:
        feature_type_norm = str(feature_type).lower().strip()
        allowed_feature_types = {"const_original", "const_unified", "random", "informative"}

        if feature_type_norm not in allowed_feature_types:
            raise ValueError(
                f"feature_type='{feature_type_norm}' not valid. "
                f"Admitted: ['const_original', 'const_unified', 'informative', 'random']"
            )

        build_lhs_embedding = (feature_type_norm == "const_original")

        super().__init__(
            data,
            col_stats_dict,
            rhs_emb_mode,
            dst_entity_table,
            num_nodes,
            embedding_dim,
            num_src_nodes,
            src_entity_table,
            build_lhs_embedding=build_lhs_embedding
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.channels = channels
        self.is_static = is_static

        self.feature_type = feature_type_norm

        self.src_entity_table_name = src_entity_table
        self.dst_entity_table_name = dst_entity_table

        if self.feature_type == "const_original":
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

            self.src_dense_encoder = None
            self.dst_dense_encoder = None
        else:
            self.encoder = None

            src_feat_dim = int(data[src_entity_table].x_dense.size(-1))
            dst_feat_dim = int(data[dst_entity_table].x_dense.size(-1))

            self.src_dense_encoder = torch.nn.Linear(src_feat_dim, channels)
            self.dst_dense_encoder = torch.nn.Linear(dst_feat_dim, channels)

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
        self.lin_offset_idgnn = torch.nn.Linear(embedding_dim, 1)
        self.lin_offset_embgnn = torch.nn.Linear(embedding_dim, 1)

        self.reset_parameters()
        
        self.gnn.to(self.device)
        self.lhs_projector.to(self.device)
        self.rhs_embedding.to(self.device)
        self.head.to(self.device)
        self.lin_offset_idgnn.to(self.device)
        self.lin_offset_embgnn.to(self.device)
        self.id_awareness_emb.to(self.device)
        self.temporal_encoder.to(self.device)

        if self.feature_type == "const_original":
            self.encoder.to(self.device)
        else:
            self.src_dense_encoder.to(self.device)
            self.dst_dense_encoder.to(self.device)
        
        if self.lhs_embedding is not None:
            self.lhs_embedding.to(self.device)
        
        self._monitor = {}
        self._last_local_assign = None

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def reset_parameters(self) -> None:
        super().reset_parameters()

        if self.feature_type == "const_original":
            self.encoder.reset_parameters()
        else:
            self.src_dense_encoder.reset_parameters()
            self.dst_dense_encoder.reset_parameters()

        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()
        self.id_awareness_emb.reset_parameters()
        self.lin_offset_embgnn.reset_parameters()
        self.lin_offset_idgnn.reset_parameters()
        self.lhs_projector.reset_parameters()

        if self.feature_type == "const_original":
            self.rhs_embedding.reset_parameters()

            if self.lhs_embedding is not None:
                self.lhs_embedding.reset_parameters()

    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType,
        dst_table: NodeType,
        score_mode: str = "full"
    ) -> Tensor:
        score_mode = str(score_mode).lower().strip()

        seed_time = batch[entity_table].seed_time

        if self.feature_type == "const_original":
            x_dict = self.encoder(batch.tf_dict)

            if self.lhs_embedding is not None:
                lhs = self.lhs_embedding()
                x_dict[entity_table] = lhs[batch[entity_table].n_id]
        elif self.feature_type in {"const_unified", "random", "informative"}:
            x_dict = {
                entity_table: self.src_dense_encoder(batch[entity_table].x_dense),
                dst_table: self.dst_dense_encoder(batch[dst_table].x_dense)
            }
        else:
            raise ValueError(f"Unknown feature_type='{self.feature_type}'")

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

        embgnn_offset_logits = self.lin_offset_embgnn(lhs_embedding_projected).flatten()

        embgnn_logits += embgnn_offset_logits.view(-1, 1)

        embgnn_logits_pre_mean = embgnn_logits.mean()
        embgnn_logits_pre_std = embgnn_logits.std()

        emb_selected_pre = embgnn_logits[lhs_idgnn_batch, rhs_idgnn_index]

        idgnn_logits = self.head(rhs_gnn_embedding).flatten()
        idgnn_logits += (lhs_embedding[lhs_idgnn_batch] * rhs_gnn_embedding).sum(dim=-1)

        idgnn_offset_logits = self.lin_offset_idgnn(lhs_embedding_projected).flatten()

        idgnn_logits = idgnn_logits + idgnn_offset_logits[lhs_idgnn_batch]

        delta_override = (idgnn_logits - emb_selected_pre)
        delta_override_mean = delta_override.mean()
        delta_override_std = delta_override.std()

        num_items = rhs_embedding.size(0)
        num_overrides = rhs_idgnn_index.numel()
        override_ratio = num_overrides / (batch_size * num_items)

        local_per_user = torch.bincount(lhs_idgnn_batch, minlength=batch_size).float()
        local_per_user_mean = local_per_user.mean()
        local_per_user_std = local_per_user.std(unbiased=False)

        if score_mode == "full":
            embgnn_logits[lhs_idgnn_batch, rhs_idgnn_index] = idgnn_logits
        elif score_mode == "global":
            pass
        elif score_mode == "local":
            local_only = torch.full(
                (batch_size, rhs_embedding.size(0)),
                float("-inf"),
                device=embgnn_logits.device,
                dtype=embgnn_logits.dtype
            )

            local_only[lhs_idgnn_batch, rhs_idgnn_index] = idgnn_logits
            embgnn_logits = local_only
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

                offset_emb_mean = embgnn_offset_logits.mean()
                offset_emb_std = embgnn_offset_logits.std()
                offset_id_mean = idgnn_offset_logits.mean()
                offset_id_std = idgnn_offset_logits.std()

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
                    "offset_emb_mean": offset_emb_mean.item(),
                    "offset_emb_std": offset_emb_std.item(),
                    "offset_id_mean": offset_id_mean.item(),
                    "offset_id_std": offset_id_std.item(),
                    "emb_logit_pre_mean": embgnn_logits_pre_mean.item(),
                    "emb_logit_pre_std": embgnn_logits_pre_std.item(),
                    "override_ratio": float(override_ratio),
                    "num_overrides": int(num_overrides),
                    "num_items": int(num_items),
                    "local_items_per_user_mean": local_per_user_mean.item(),
                    "local_items_per_user_std": local_per_user_std.item(),
                    "delta_override_mean": delta_override_mean.item(),
                    "delta_override_std": delta_override_std.item()
                }
        
        if not self.training:
            self._last_local_assign = (
                lhs_idgnn_batch.detach().cpu(),
                rhs_idgnn_index.detach().cpu(),
                batch_size
            )
        else:
            self._last_local_assign = None

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