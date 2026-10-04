from __future__ import annotations
from typing import Dict, List

import torch
from torch import Tensor
from torch_geometric.nn import HeteroConv
from torch_geometric.typing import EdgeType, NodeType

from torch_sparse import SparseTensor

from .LightGCNLayer import LGConv, to_sparse_adj_t, normalize_adj_t_bipartite

class HeteroLightGCN(torch.nn.Module):
    def __init__(
        self,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        aggr: str = "add",
        normalize: bool = True,
        num_layers: int = 2
    ) -> None:
        super().__init__()

        self.normalize = bool(normalize)

        self.convs = torch.nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(HeteroConv(
                {
                    edge_type: LGConv(aggr=aggr, normalize=False) for edge_type in edge_types
                },
                aggr="sum"
            ))

        self.node_types = list(node_types)
        self.edge_types = list(edge_types)

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()

    @torch.no_grad()
    def _build_adj_t_dict(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, object]
    ) -> Dict[EdgeType, SparseTensor]:
        adj_t_dict: Dict[EdgeType, SparseTensor] = {}

        for edge_type, adj in edge_index_dict.items():
            src_type, _, dst_type = edge_type

            num_src = x_dict[src_type].size(0)
            num_dst = x_dict[dst_type].size(0)

            adj_t = to_sparse_adj_t(adj, num_src=num_src, num_dst=num_dst)

            if self.normalize:
                adj_t = normalize_adj_t_bipartite(adj_t)

            adj_t_dict[edge_type] = adj_t

        return adj_t_dict

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, object]
    ) -> Dict[NodeType, Tensor]:
        adj_t_dict = self._build_adj_t_dict(x_dict, edge_index_dict)

        for conv in self.convs:
            x_dict = conv(x_dict, adj_t_dict)

        return x_dict