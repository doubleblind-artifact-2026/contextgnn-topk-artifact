"""
This file is adapted from code originally released in the ContextGNN project.

Original source:
- https://github.com/kumo-ai/ContextGNN

The original code is distributed under the MIT License. The corresponding
copyright and license notices are preserved in this repository.

Modifications were made for the experiments accompanying this submission.
"""

from typing import Any, Dict, Optional

import torch
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from typing_extensions import Self

from ..encoder import DEFAULT_STYPE_ENCODER_DICT
from ..rhs_embedding import RHSEmbedding
from ...utils import RHSEmbeddingMode

class RHSEmbeddingGNN(torch.nn.Module):
    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        rhs_emb_mode: RHSEmbeddingMode,
        dst_entity_table: str,
        num_nodes: int,
        embedding_dim: int,
        num_src_nodes: Optional[int] = None,
        src_entity_table: Optional[str] = None,
        build_lhs_embedding: bool = True
    ):
        super().__init__()

        def build_rhs_module(node_table: str, n_nodes: int) -> RHSEmbedding:
            store = data[node_table]

            if "tf" in store.keys():
                stype_encoder_dict = {
                    k: v[0]()
                    for k, v in DEFAULT_STYPE_ENCODER_DICT.items()
                    if k in store["tf"].col_names_dict.keys()
                }

                return RHSEmbedding(
                    emb_mode=rhs_emb_mode,
                    embedding_dim=embedding_dim,
                    num_nodes=n_nodes,
                    col_stats=col_stats_dict[node_table],
                    col_names_dict=store["tf"].col_names_dict,
                    stype_encoder_dict=stype_encoder_dict,
                    feat=store["tf"],
                    dense_feat=None
                )

            if "x_dense" in store.keys():
                return RHSEmbedding(
                    emb_mode=rhs_emb_mode,
                    embedding_dim=embedding_dim,
                    num_nodes=n_nodes,
                    col_stats=None,
                    col_names_dict=None,
                    stype_encoder_dict=None,
                    feat=None,
                    dense_feat=store["x_dense"]
                )

            raise KeyError(f"Neither 'tf' nor 'x_dense' found for node table '{node_table}'")

        self.rhs_embedding = build_rhs_module(dst_entity_table, num_nodes)
        self.lhs_embedding = None

        if build_lhs_embedding and num_src_nodes is not None:
            assert src_entity_table is not None
            self.lhs_embedding = build_rhs_module(src_entity_table, num_src_nodes)

    def reset_parameters(self):
        self.rhs_embedding.reset_parameters()

        if self.lhs_embedding is not None:
            self.lhs_embedding.reset_parameters()

    def to(self, *args, **kwargs) -> Self:
        self.rhs_embedding.to(*args, **kwargs)

        if self.lhs_embedding is not None:
            self.lhs_embedding.to(*args, **kwargs)

        return super().to(*args, **kwargs)

    def cpu(self) -> Self:
        self.rhs_embedding.cpu()

        if self.lhs_embedding is not None:
            self.lhs_embedding.cpu()

        return super().cpu()

    def cuda(self, *args, **kwargs) -> Self:
        self.rhs_embedding.cuda(*args, **kwargs)

        if self.lhs_embedding is not None:
            self.lhs_embedding.cuda(*args, **kwargs)

        return super().cuda(*args, **kwargs)