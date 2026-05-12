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
        cl_method: str = "none",
        lambda_cl: float = 0.0,
        tau_cl: float = 0.2,
        edge_drop_p: float = 0.1,
        xsim_eps: float = 0.1,
        cl_layer: int = 1,
        cl_scope: str = "user_item",
        cl_normalize: bool = True
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
        self.lin_offset_idgnn = torch.nn.Linear(embedding_dim, 1)
        self.lin_offset_embgnn = torch.nn.Linear(embedding_dim, 1)
        self.channels = channels
        self.is_static = is_static
        self.reset_parameters()
        
        self.gnn.to(self.device)
        self.lhs_projector.to(self.device)
        self.rhs_embedding.to(self.device)
        self.head.to(self.device)
        self.lin_offset_idgnn.to(self.device)
        self.lin_offset_embgnn.to(self.device)
        self.id_awareness_emb.to(self.device)
        
        self.encoder.to(self.device)
        self.temporal_encoder.to(self.device)
        
        if self.lhs_embedding is not None:
            self.lhs_embedding.to(self.device)
        
        self.cl_method = cl_method
        self.lambda_cl = lambda_cl
        self.tau_cl = tau_cl
        self.edge_drop_p = edge_drop_p
        self.xsim_eps = xsim_eps
        self.cl_layer = cl_layer
        self.cl_scope = cl_scope
        self.cl_normalize = cl_normalize
        
        self._monitor = {}
        self._last_local_assign = None

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def reset_parameters(self) -> None:
        super().reset_parameters()

        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()
        self.id_awareness_emb.reset_parameters()
        self.rhs_embedding.reset_parameters()
        self.lin_offset_embgnn.reset_parameters()
        self.lin_offset_idgnn.reset_parameters()
        self.lhs_projector.reset_parameters()

        if self.lhs_embedding is not None:
            self.lhs_embedding.reset_parameters()

    def _encode_batch(
        self,
        batch: HeteroData,
        entity_table: NodeType,
        dst_table: NodeType,
        edge_index_dict_override=None,
        x_dict_override=None,
        return_all_layers: bool = False,
        perturbed: bool = False
    ):
        seed_time = batch[entity_table].seed_time

        if x_dict_override is None:
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
        else:
            x_dict = x_dict_override

        edge_index_dict = edge_index_dict_override if edge_index_dict_override is not None else batch.edge_index_dict

        if return_all_layers:
            if not hasattr(self.gnn, "forward_with_all_layers"):
                raise NotImplementedError(
                    "This backbone does not expose forward_with_all_layers(...). "
                    "XSimGCL requires layer-wise outputs."
                )

            x_out_dict, layer_outputs = self.gnn.forward_with_all_layers(
                x_dict,
                edge_index_dict,
                perturbed=perturbed,
                eps=(self.xsim_eps if perturbed else 0.0)
            )
        else:
            if perturbed:
                raise NotImplementedError(
                    "Perturbed forward requires return_all_layers=True and a backbone "
                    "supporting forward_with_all_layers(...)."
                )

            x_out_dict = self.gnn(x_dict, edge_index_dict)
            layer_outputs = None

        batch_size = seed_time.size(0)

        lhs_embedding = x_out_dict[entity_table][:batch_size]
        lhs_embedding_projected = self.lhs_projector(lhs_embedding)
        rhs_gnn_embedding = x_out_dict[dst_table]
        rhs_idgnn_index = batch.n_id_dict[dst_table]
        lhs_idgnn_batch = batch.batch_dict[dst_table]
        rhs_embedding = self.rhs_embedding()

        out = {
            "batch_size": batch_size,
            "lhs_embedding": lhs_embedding,
            "lhs_embedding_projected": lhs_embedding_projected,
            "rhs_gnn_embedding": rhs_gnn_embedding,
            "rhs_idgnn_index": rhs_idgnn_index,
            "lhs_idgnn_batch": lhs_idgnn_batch,
            "rhs_embedding": rhs_embedding,
            "x_out_dict": x_out_dict
        }

        if return_all_layers:
            out["layer_outputs"] = layer_outputs

        return out

    def _compute_logits_from_encoded(self, encoded, score_mode: str = "full"):
        score_mode = str(score_mode).lower().strip()

        lhs_embedding = encoded["lhs_embedding"]
        lhs_embedding_projected = encoded["lhs_embedding_projected"]
        rhs_gnn_embedding = encoded["rhs_gnn_embedding"]
        rhs_idgnn_index = encoded["rhs_idgnn_index"]
        lhs_idgnn_batch = encoded["lhs_idgnn_batch"]
        rhs_embedding = encoded["rhs_embedding"]
        batch_size = encoded["batch_size"]

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

        monitor = {
            "embgnn_offset_logits": embgnn_offset_logits,
            "idgnn_offset_logits": idgnn_offset_logits,
            "embgnn_logits_pre_mean": embgnn_logits_pre_mean,
            "embgnn_logits_pre_std": embgnn_logits_pre_std,
            "idgnn_logits": idgnn_logits,
            "override_ratio": float(override_ratio),
            "num_overrides": int(num_overrides),
            "num_items": int(num_items),
            "local_items_per_user_mean": local_per_user_mean.item(),
            "local_items_per_user_std": local_per_user_std.item(),
            "delta_override_mean": delta_override_mean.item(),
            "delta_override_std": delta_override_std.item()
        }

        return embgnn_logits, monitor

    def _update_training_monitor(self, encoded, logits, score_monitor):
        with torch.no_grad():
            lhs_embedding = encoded["lhs_embedding"]
            rhs_gnn_embedding = encoded["rhs_gnn_embedding"]
            rhs_embedding = encoded["rhs_embedding"]

            lhs_norm = lhs_embedding.norm(p=2, dim=1)
            rhs_gnn_norm = rhs_gnn_embedding.norm(p=2, dim=1)
            rhs_emb_norm = rhs_embedding.norm(p=2, dim=1)

            lhs_disp = lhs_embedding.var(dim=0, unbiased=False).mean()
            rhs_gnn_disp = rhs_gnn_embedding.var(dim=0, unbiased=False).mean()
            rhs_emb_disp = rhs_embedding.var(dim=0, unbiased=False).mean()

            emb_logits_mean = logits.mean()
            emb_logits_std = logits.std()
            id_logits_mean = score_monitor["idgnn_logits"].mean()
            id_logits_std = score_monitor["idgnn_logits"].std()

            offset_emb_logits = score_monitor["embgnn_offset_logits"]
            offset_id_logits = score_monitor["idgnn_offset_logits"]

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
                "offset_emb_mean": offset_emb_logits.mean().item(),
                "offset_emb_std": offset_emb_logits.std().item(),
                "offset_id_mean": offset_id_logits.mean().item(),
                "offset_id_std": offset_id_logits.std().item(),
                "emb_logit_pre_mean": score_monitor["embgnn_logits_pre_mean"].item(),
                "emb_logit_pre_std": score_monitor["embgnn_logits_pre_std"].item(),
                "override_ratio": score_monitor["override_ratio"],
                "num_overrides": score_monitor["num_overrides"],
                "num_items": score_monitor["num_items"],
                "local_items_per_user_mean": score_monitor["local_items_per_user_mean"],
                "local_items_per_user_std": score_monitor["local_items_per_user_std"],
                "delta_override_mean": score_monitor["delta_override_mean"],
                "delta_override_std": score_monitor["delta_override_std"]
            }

    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType,
        dst_table: NodeType,
        score_mode: str = "full"
    ) -> Tensor:
        encoded = self._encode_batch(
            batch=batch,
            entity_table=entity_table,
            dst_table=dst_table
        )

        logits, score_monitor = self._compute_logits_from_encoded(encoded, score_mode=score_mode)

        if self.training:
            self._update_training_monitor(encoded, logits, score_monitor)

        if not self.training:
            self._last_local_assign = (
                encoded["lhs_idgnn_batch"].detach().cpu(),
                encoded["rhs_idgnn_index"].detach().cpu(),
                encoded["batch_size"]
            )
        else:
            self._last_local_assign = None

        return logits

    def forward_with_cl(
        self,
        batch: HeteroData,
        entity_table: NodeType,
        dst_table: NodeType,
        score_mode: str = "full",
        cl_override_payload=None,
        cl_indices=None
    ):
        cl_payload = None

        xsimgcl_active = (
            self.training and
            self.cl_method == "xsimgcl" and
            self.lambda_cl > 0.0
        )

        if xsimgcl_active:
            cl_payload = self._build_xsimgcl_payload(
                batch=batch,
                entity_table=entity_table,
                dst_table=dst_table,
                cl_indices=cl_indices
            )

            encoded_for_supervision = cl_payload.pop("encoded_for_supervision")

            logits, score_monitor = self._compute_logits_from_encoded(
                encoded_for_supervision,
                score_mode=score_mode
            )

            self._update_training_monitor(encoded_for_supervision, logits, score_monitor)

            return {
                "logits": logits,
                "encoded_supervision": encoded_for_supervision,
                "cl_payload": cl_payload
            }

        encoded_clean = self._encode_batch(
            batch=batch,
            entity_table=entity_table,
            dst_table=dst_table
        )

        logits, score_monitor = self._compute_logits_from_encoded(
            encoded_clean,
            score_mode=score_mode
        )

        if self.training:
            self._update_training_monitor(encoded_clean, logits, score_monitor)

        if self.training and self.cl_method != "none" and self.lambda_cl > 0.0:
            if self.cl_method == "sgl":
                if cl_override_payload is None:
                    raise ValueError(
                        "SGL requires cl_override_payload prepared in ContextGNN.py "
                        "before sparse adjacency conversion."
                    )

                cl_payload = self._build_sgl_payload(
                    batch,
                    entity_table,
                    dst_table,
                    cl_override_payload=cl_override_payload,
                    cl_indices=cl_indices
                )

        return {
            "logits": logits,
            "encoded_clean": encoded_clean,
            "cl_payload": cl_payload
        }

    def to(self, *args, **kwargs) -> Self:
        return super().to(*args, **kwargs)

    def cpu(self) -> Self:
        return super().cpu()

    def cuda(self, *args, **kwargs) -> Self:
        return super().cuda(*args, **kwargs)

    def _maybe_normalize(self, x: Tensor) -> Tensor:
        if self.cl_normalize:
            return torch.nn.functional.normalize(x, dim=-1)
        
        return x

    def _info_nce_loss(self, z1: Tensor, z2: Tensor, tau: float):
        if z1.size(0) == 0 or z2.size(0) == 0:
            zero = z1.new_tensor(0.0)
            empty_logits = z1.new_empty((0, 0))
            
            return zero, empty_logits

        z1 = self._maybe_normalize(z1)
        z2 = self._maybe_normalize(z2)

        logits = torch.matmul(z1, z2.t()) / tau
        labels = torch.arange(z1.size(0), device=z1.device)
        loss = torch.nn.functional.cross_entropy(logits, labels)

        return loss, logits

    def _uniformity(self, z: Tensor):
        z = self._maybe_normalize(z)

        if z.size(0) < 2:
            return z.new_tensor(0.0)
        
        pdist = torch.pdist(z, p=2).pow(2)

        return torch.log(torch.exp(-2.0 * pdist).mean() + 1e-12)

    def _embedding_std_per_dim_mean(self, z: Tensor):
        if z.size(0) < 2:
            return z.new_tensor(0.0)
        
        return z.std(dim=0, unbiased=False).mean()
    
    def _pair_contrastive_stats(self, view1: Tensor, view2: Tensor):
        if view1.size(0) == 0 or view2.size(0) == 0:
            ref = view1 if view1.numel() > 0 else view2
            zero = ref.new_tensor(0.0)

            return {
                "pos_sim_mean": zero,
                "pos_sim_std": zero,
                "neg_sim_mean": zero,
                "neg_sim_std": zero,
                "hardest_neg_sim_mean": zero,
                "alignment_mean": zero,
                "uniformity_mean": zero
            }

        z1 = self._maybe_normalize(view1)
        z2 = self._maybe_normalize(view2)

        pos_sim = (z1 * z2).sum(dim=-1)

        sim_mat = torch.matmul(z1, z2.t())
        n = sim_mat.size(0)

        if n > 1:
            neg_mask = ~torch.eye(n, dtype=torch.bool, device=sim_mat.device)
            neg_vals = sim_mat[neg_mask]
            neg_sim_mean = neg_vals.mean()
            neg_sim_std = neg_vals.std(unbiased=False)

            sim_mat_masked = sim_mat.masked_fill(~neg_mask, float("-inf"))
            hardest_neg_sim_mean = sim_mat_masked.max(dim=1).values.mean()
        else:
            neg_sim_mean = sim_mat.new_tensor(0.0)
            neg_sim_std = sim_mat.new_tensor(0.0)
            hardest_neg_sim_mean = sim_mat.new_tensor(0.0)

        alignment_mean = ((z1 - z2).norm(p=2, dim=-1).pow(2)).mean()
        uniformity_mean = 0.5 * (self._uniformity(view1) + self._uniformity(view2))

        return {
            "pos_sim_mean": pos_sim.mean(),
            "pos_sim_std": pos_sim.std(unbiased=False),
            "neg_sim_mean": neg_sim_mean,
            "neg_sim_std": neg_sim_std,
            "hardest_neg_sim_mean": hardest_neg_sim_mean,
            "alignment_mean": alignment_mean,
            "uniformity_mean": uniformity_mean
        }

    def _combine_user_item_stats(
        self,
        user_stats: dict,
        item_stats: dict,
        user_count: int,
        item_count: int
    ):
        if self.cl_scope == "user":
            return {k: v.item() for k, v in user_stats.items()}

        if self.cl_scope == "item":
            return {k: v.item() for k, v in item_stats.items()}

        total = int(user_count) + int(item_count)

        if total <= 0:
            return {k: float("nan") for k in user_stats.keys()}

        combined = {}

        for key in user_stats.keys():
            combined[key] = ((user_stats[key] * float(user_count) + item_stats[key] * float(item_count)) / float(total)).item()

        return combined
    
    def _map_global_items_to_local_idx(
        self,
        rhs_global_idx: Tensor,
        pos_item_global_idx: Tensor
    ) -> Tensor:
        if rhs_global_idx.numel() == 0 or pos_item_global_idx.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=rhs_global_idx.device)

        rhs_map = {
            int(g): i for i, g in enumerate(rhs_global_idx.detach().cpu().tolist())
        }

        local_idx = [
            rhs_map[int(g)]
            for g in torch.unique(pos_item_global_idx).detach().cpu().tolist()
            if int(g) in rhs_map
        ]

        if len(local_idx) == 0:
            return torch.empty(0, dtype=torch.long, device=rhs_global_idx.device)

        return torch.tensor(local_idx, dtype=torch.long, device=rhs_global_idx.device)

    def _build_sgl_payload(self, batch, entity_table, dst_table, cl_override_payload, cl_indices):
        if cl_indices is None:
            raise ValueError(
                "SGL requires cl_indices so that user/item anchors are aligned "
                "with the supervised batch and deduplicated consistently."
            )

        enc_view1 = self._encode_batch(
            batch=batch,
            entity_table=entity_table,
            dst_table=dst_table,
            edge_index_dict_override=cl_override_payload["view1_edge_index_dict"]
        )

        enc_view2 = self._encode_batch(
            batch=batch,
            entity_table=entity_table,
            dst_table=dst_table,
            edge_index_dict_override=cl_override_payload["view2_edge_index_dict"]
        )

        user_local_idx = torch.unique(cl_indices["user_local_idx"]).long()

        pos_item_global_idx = torch.unique(cl_indices["pos_item_global_idx"]).long()

        item_local_idx = self._map_global_items_to_local_idx(
            enc_view1["rhs_idgnn_index"],
            pos_item_global_idx
        )

        num_dedup_pos_items = int(pos_item_global_idx.numel())

        return {
            "method": "sgl",
            "user_view1": enc_view1["lhs_embedding"][user_local_idx],
            "user_view2": enc_view2["lhs_embedding"][user_local_idx],
            "item_view1": enc_view1["rhs_gnn_embedding"][item_local_idx],
            "item_view2": enc_view2["rhs_gnn_embedding"][item_local_idx],
            "sgl_stats": cl_override_payload["sgl_stats"],
            "num_dedup_pos_items": num_dedup_pos_items
        }
    
    def _xsimgcl_side_stats(
        self,
        clean_layer_view: Tensor,
        pert_layer_view: Tensor,
        final_view: Tensor,
        cl_view: Tensor
    ):
        if clean_layer_view.size(0) == 0:
            ref = final_view if final_view.numel() > 0 else pert_layer_view
            zero = ref.new_tensor(0.0)

            return {
                "noise_norm_mean": zero,
                "noise_norm_std": zero,
                "noise_to_signal_ratio_mean": zero,
                "clean_noisy_cos_mean": zero,
                "cl_layer_pos_sim_mean": zero,
                "cl_layer_neg_sim_mean": zero
            }

        noise = pert_layer_view - clean_layer_view
        noise_norm = noise.norm(p=2, dim=-1)
        signal_norm = clean_layer_view.norm(p=2, dim=-1)

        clean_noisy_cos = torch.nn.functional.cosine_similarity(
            clean_layer_view,
            pert_layer_view,
            dim=-1
        )

        pair_stats = self._pair_contrastive_stats(final_view, cl_view)

        return {
            "noise_norm_mean": noise_norm.mean(),
            "noise_norm_std": noise_norm.std(unbiased=False),
            "noise_to_signal_ratio_mean": (noise_norm / (signal_norm + 1e-12)).mean(),
            "clean_noisy_cos_mean": clean_noisy_cos.mean(),
            "cl_layer_pos_sim_mean": pair_stats["pos_sim_mean"],
            "cl_layer_neg_sim_mean": pair_stats["neg_sim_mean"]
        }
    
    def _build_xsimgcl_stats(
        self,
        user_clean_layer_view: Tensor,
        user_pert_layer_view: Tensor,
        user_final_view: Tensor,
        user_cl_view: Tensor,
        item_clean_layer_view: Tensor,
        item_pert_layer_view: Tensor,
        item_final_view: Tensor,
        item_cl_view: Tensor
    ):
        user_stats = self._xsimgcl_side_stats(
            user_clean_layer_view,
            user_pert_layer_view,
            user_final_view,
            user_cl_view
        )

        item_stats = self._xsimgcl_side_stats(
            item_clean_layer_view,
            item_pert_layer_view,
            item_final_view,
            item_cl_view
        )

        combined = self._combine_user_item_stats(
            user_stats,
            item_stats,
            user_count=int(user_final_view.size(0)),
            item_count=int(item_final_view.size(0))
        )

        combined["cl_layer"] = float(self.cl_layer)

        return combined
    
    def _build_xsimgcl_payload(
        self,
        batch,
        entity_table,
        dst_table,
        cl_indices
    ):
        if cl_indices is None:
            raise ValueError("XSimGCL requires cl_indices with batch users and positive items.")

        clean_encoded = self._encode_batch(
            batch=batch,
            entity_table=entity_table,
            dst_table=dst_table,
            return_all_layers=True,
            perturbed=False
        )

        pert_encoded = self._encode_batch(
            batch=batch,
            entity_table=entity_table,
            dst_table=dst_table,
            return_all_layers=True,
            perturbed=True
        )

        clean_layer_outputs = clean_encoded.get("layer_outputs", None)
        pert_layer_outputs = pert_encoded.get("layer_outputs", None)

        if clean_layer_outputs is None or pert_layer_outputs is None:
            raise NotImplementedError("XSimGCL requires layer-wise outputs from the selected backbone.")

        if self.cl_layer > len(pert_layer_outputs):
            raise ValueError(f"cl_layer={self.cl_layer} exceeds available GNN layers={len(pert_layer_outputs)}")

        layer_idx = self.cl_layer - 1
        batch_size = pert_encoded["batch_size"]

        user_local_idx = torch.unique(cl_indices["user_local_idx"]).long()
        pos_item_global_idx = torch.unique(cl_indices["pos_item_global_idx"]).long()

        item_local_idx = self._map_global_items_to_local_idx(
            pert_encoded["rhs_idgnn_index"],
            pos_item_global_idx
        )

        num_dedup_pos_items = int(pos_item_global_idx.numel())

        clean_user_layer_all = clean_layer_outputs[layer_idx][entity_table][:batch_size]
        pert_user_layer_all = pert_layer_outputs[layer_idx][entity_table][:batch_size]

        clean_item_layer_all = clean_layer_outputs[layer_idx][dst_table]
        pert_item_layer_all = pert_layer_outputs[layer_idx][dst_table]

        user_final_view = pert_encoded["lhs_embedding"][user_local_idx]
        user_cl_view = pert_user_layer_all[user_local_idx]

        item_final_view = pert_encoded["rhs_gnn_embedding"][item_local_idx]
        item_cl_view = pert_item_layer_all[item_local_idx]

        user_clean_layer_view = clean_user_layer_all[user_local_idx]
        user_pert_layer_view = pert_user_layer_all[user_local_idx]

        item_clean_layer_view = clean_item_layer_all[item_local_idx]
        item_pert_layer_view = pert_item_layer_all[item_local_idx]

        xsimgcl_stats = self._build_xsimgcl_stats(
            user_clean_layer_view=user_clean_layer_view,
            user_pert_layer_view=user_pert_layer_view,
            user_final_view=user_final_view,
            user_cl_view=user_cl_view,
            item_clean_layer_view=item_clean_layer_view,
            item_pert_layer_view=item_pert_layer_view,
            item_final_view=item_final_view,
            item_cl_view=item_cl_view
        )

        return {
            "method": "xsimgcl",
            "user_view1": user_final_view,
            "user_view2": user_cl_view,
            "item_view1": item_final_view,
            "item_view2": item_cl_view,
            "xsimgcl_stats": xsimgcl_stats,
            "encoded_for_supervision": pert_encoded,
            "num_dedup_pos_items": num_dedup_pos_items
        }

    def compute_contrastive_loss(self, cl_payload):
        method = cl_payload["method"]

        user_view1 = cl_payload["user_view1"]
        user_view2 = cl_payload["user_view2"]
        item_view1 = cl_payload["item_view1"]
        item_view2 = cl_payload["item_view2"]

        user_loss = torch.zeros((), device=user_view1.device)
        item_loss = torch.zeros((), device=user_view1.device)

        if self.cl_scope in {"user", "user_item"}:
            user_loss, user_logits = self._info_nce_loss(user_view1, user_view2, self.tau_cl)

        if self.cl_scope in {"item", "user_item"}:
            item_loss, item_logits = self._info_nce_loss(item_view1, item_view2, self.tau_cl)

        active_losses = []

        if self.cl_scope in {"user", "user_item"} and user_view1.size(0) > 0:
            active_losses.append(user_loss)

        if self.cl_scope in {"item", "user_item"} and item_view1.size(0) > 0:
            active_losses.append(item_loss)

        if len(active_losses) == 0:
            ref_device = user_view1.device if user_view1.numel() > 0 else item_view1.device
            loss_cl = torch.zeros((), device=ref_device)
        else:
            loss_cl = torch.stack(active_losses).mean()

        with torch.no_grad():
            user_stats = self._pair_contrastive_stats(user_view1, user_view2)
            item_stats = self._pair_contrastive_stats(item_view1, item_view2)

            user_count = int(user_view1.size(0)) if self.cl_scope in {"user", "user_item"} else 0
            item_count = int(item_view1.size(0)) if self.cl_scope in {"item", "user_item"} else 0

            item_side_enabled = self.cl_scope in {"item", "user_item"}
            num_dedup_pos_items = int(cl_payload.get("num_dedup_pos_items", item_count))

            if item_side_enabled:
                item_cl_coverage = (
                    float(item_count) / float(num_dedup_pos_items)
                    if num_dedup_pos_items > 0 else 0.0
                )

                empty_item_cl_batch = 1.0 if item_count == 0 else 0.0
            else:
                item_cl_coverage = 0.0
                empty_item_cl_batch = 0.0

            common_stats = self._combine_user_item_stats(
                user_stats,
                item_stats,
                user_count=user_count,
                item_count=item_count
            )

            user_emb_std_per_dim_mean = self._embedding_std_per_dim_mean(user_view1)
            item_emb_std_per_dim_mean = self._embedding_std_per_dim_mean(item_view1)

            monitor = {
                **common_stats,
                "user_emb_std_per_dim_mean": user_emb_std_per_dim_mean.item(),
                "item_emb_std_per_dim_mean": item_emb_std_per_dim_mean.item(),
                "user_cl_count": user_count,
                "item_cl_count": item_count,
                "cl_total_count": user_count + item_count,
                "item_cl_coverage": item_cl_coverage,
                "empty_item_cl_batch": empty_item_cl_batch
            }

            if method == "sgl":
                sgl_stats = cl_payload["sgl_stats"]

                monitor.update({
                    "edge_drop_rate_view1": sgl_stats["edge_drop_rate_view1"],
                    "edge_drop_rate_view2": sgl_stats["edge_drop_rate_view2"],
                    "kept_edges_view1": sgl_stats["kept_edges_view1"],
                    "kept_edges_view2": sgl_stats["kept_edges_view2"],
                    "retained_edge_overlap_ratio": sgl_stats["retained_edge_overlap_ratio"],
                    "isolated_users_ratio_view1": sgl_stats["isolated_users_ratio_view1"],
                    "isolated_users_ratio_view2": sgl_stats["isolated_users_ratio_view2"],
                    "sgl_message_edges_count": sgl_stats["sgl_message_edges_count"],
                    "sgl_seed_users_count": sgl_stats["sgl_seed_users_count"]
                })
            elif method == "xsimgcl":
                monitor.update(cl_payload["xsimgcl_stats"])

        return loss_cl, monitor

    def get_top_k(self, preds, train_mask, k=100):
        if not torch.is_tensor(preds):
            preds = torch.as_tensor(preds)

        if preds.device != self.device:
            preds = preds.to(self.device, non_blocking=True)

        mask = torch.as_tensor(train_mask, device=preds.device, dtype=torch.bool)
        preds.masked_fill_(~mask, float("-inf"))

        return torch.topk(preds, k=k, dim=1, sorted=True)