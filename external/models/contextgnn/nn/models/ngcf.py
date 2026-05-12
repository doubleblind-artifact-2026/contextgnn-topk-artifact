from __future__ import annotations
from typing import Dict, List

import torch
from torch import Tensor
from torch_geometric.nn import HeteroConv, LayerNorm
from torch_geometric.typing import EdgeType, NodeType
from torch_sparse import SparseTensor

from .NGCFLayer import NGCFLayer, to_sparse_adj_t, normalize_adj_t_bipartite

class HeteroNGCF(torch.nn.Module):
    def __init__(
        self,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        channels: int,
        aggr: str = "add",
        dropout: float = 0.5,
        num_layers: int = 2,
        normalize: bool = True
    ) -> None:
        super().__init__()

        self.normalize = bool(normalize)

        self.convs = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: NGCFLayer(channels, channels, aggr=aggr) for edge_type in edge_types
                },
                aggr="sum"
            )

            self.convs.append(conv)
            self.dropouts.append(torch.nn.Dropout(p=dropout))

            norm_dict = torch.nn.ModuleDict()

            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            
            self.norms.append(norm_dict)

        self.node_types = list(node_types)
        self.edge_types = list(edge_types)

        self.channels = channels

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()

        for drop in self.dropouts:
            _ = drop

        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

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

        for conv, dropout, norm_dict in zip(self.convs, self.dropouts, self.norms):
            x_dict = conv(x_dict, adj_t_dict)
            x_dict = {k: dropout(v) for k, v in x_dict.items()}
            x_dict = {k: norm_dict[k](v) for k, v in x_dict.items()}
            x_dict = {k: v.relu() for k, v in x_dict.items()}

        return x_dict