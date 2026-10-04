from __future__ import annotations
from typing import Optional, Tuple, Union

import torch
from torch import Tensor

from torch_sparse import SparseTensor

AdjLike = Union[Tensor, SparseTensor]

def to_sparse_adj_t(
    edge_index: AdjLike,
    num_src: int,
    num_dst: int,
    edge_weight: Optional[Tensor] = None
) -> SparseTensor:
    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index

        if adj_t.sparse_sizes() == (num_src, num_dst):
            adj_t = adj_t.t()
        elif adj_t.sparse_sizes() != (num_dst, num_src):
            raise ValueError(
                f"SparseTensor shape {adj_t.sparse_sizes()} incompatibile con "
                f"(num_dst={num_dst}, num_src={num_src}) o (num_src, num_dst)."
            )

        if not adj_t.has_value():
            nnz = adj_t.nnz()
            ones = torch.ones(nnz, device=adj_t.device(), dtype=torch.float32)
            adj_t = adj_t.set_value(ones, layout="coo")
        
        return adj_t.coalesce()
    
    if isinstance(edge_index, Tensor) and edge_index.dim() == 2 and edge_index.size(0) == 2 and edge_index.dtype == torch.long:
        src = edge_index[0]
        dst = edge_index[1]

        if edge_weight is None:
            val = torch.ones(src.numel(), device=edge_index.device, dtype=torch.float32)
        else:
            val = edge_weight.to(torch.float32)

        adj_t = SparseTensor(
            row=dst,
            col=src,
            value=val,
            sparse_sizes=(num_dst, num_src)
        )

        return adj_t.coalesce()

    if isinstance(edge_index, Tensor) and edge_index.is_sparse:
        coo = edge_index.coalesce()
        idx = coo.indices()
        val = coo.values()

        shape = tuple(coo.size())

        if shape == (num_dst, num_src):
            row = idx[0].to(torch.long)
            col = idx[1].to(torch.long)
        elif shape == (num_src, num_dst):
            row = idx[1].to(torch.long)
            col = idx[0].to(torch.long)
        else:
            raise ValueError(
                f"torch.sparse COO shape {shape} incompatibile con "
                f"(num_dst={num_dst}, num_src={num_src}) o (num_src, num_dst)."
            )

        if edge_weight is None:
            val = val.to(torch.float32)
        else:
            val = edge_weight.to(torch.float32)

        adj_t = SparseTensor(row=row, col=col, value=val, sparse_sizes=(num_dst, num_src))

        return adj_t.coalesce()

    if isinstance(edge_index, Tensor) and edge_index.layout == torch.sparse_csr:
        coo = edge_index.to_sparse_coo().coalesce()
        idx = coo.indices()
        val = coo.values()
        shape = tuple(coo.size())

        if shape == (num_dst, num_src):
            row = idx[0].to(torch.long)
            col = idx[1].to(torch.long)
        elif shape == (num_src, num_dst):
            row = idx[1].to(torch.long)
            col = idx[0].to(torch.long)
        else:
            raise ValueError(
                f"torch.sparse CSR->COO shape {shape} incompatibile con "
                f"(num_dst={num_dst}, num_src={num_src}) o (num_src, num_dst)."
            )

        if edge_weight is None:
            val = val.to(torch.float32)
        else:
            val = edge_weight.to(torch.float32)

        adj_t = SparseTensor(row=row, col=col, value=val, sparse_sizes=(num_dst, num_src))

        return adj_t.coalesce()
    
    if isinstance(edge_index, Tensor) and edge_index.dim() == 2:
        mat = edge_index
        shape = tuple(mat.size())

        if shape == (num_dst, num_src):
            nz = mat.nonzero(as_tuple=False)

            if nz.numel() == 0:
                return SparseTensor(
                    row=torch.empty(0, dtype=torch.long, device=mat.device),
                    col=torch.empty(0, dtype=torch.long, device=mat.device),
                    value=torch.empty(0, dtype=torch.float32, device=mat.device),
                    sparse_sizes=(num_dst, num_src)
                )
            
            row = nz[:, 0].to(torch.long)
            col = nz[:, 1].to(torch.long)
            val = mat[row, col].to(torch.float32)
        elif shape == (num_src, num_dst):
            nz = mat.nonzero(as_tuple=False)

            if nz.numel() == 0:
                return SparseTensor(
                    row=torch.empty(0, dtype=torch.long, device=mat.device),
                    col=torch.empty(0, dtype=torch.long, device=mat.device),
                    value=torch.empty(0, dtype=torch.float32, device=mat.device),
                    sparse_sizes=(num_dst, num_src)
                )
            
            src = nz[:, 0].to(torch.long)
            dst = nz[:, 1].to(torch.long)
            row = dst
            col = src
            val = mat[src, dst].to(torch.float32)
        else:
            raise ValueError(
                f"Dense adjacency shape {shape} incompatibile con "
                f"(num_dst={num_dst}, num_src={num_src}) o (num_src, num_dst)."
            )

        if edge_weight is not None:
            val = edge_weight.to(torch.float32)

        adj_t = SparseTensor(row=row, col=col, value=val, sparse_sizes=(num_dst, num_src))

        return adj_t.coalesce()

    raise ValueError("edge_index format not supported for LightGCNLayer.")

def normalize_adj_t_bipartite(adj_t: SparseTensor, eps: float = 1e-12) -> SparseTensor:
    adj_t = adj_t.coalesce()

    deg_dst = adj_t.sum(dim=1).to(torch.float32)
    deg_src = adj_t.sum(dim=0).to(torch.float32)

    row, col, val = adj_t.coo()

    if val is None:
        val = torch.ones(row.numel(), device=row.device, dtype=torch.float32)
    else:
        val = val.to(torch.float32)

    denom = (deg_dst[row].sqrt() * deg_src[col].sqrt()).clamp(min=eps)
    norm_val = val / denom

    return adj_t.set_value(norm_val, layout="coo").coalesce()

class LGConv(torch.nn.Module):
    def __init__(self, aggr: str = "add", normalize: bool = False):
        super().__init__()

        self.aggr = aggr
        self.normalize = normalize

    def reset_parameters(self) -> None:
        return

    def forward(
        self,
        x: Union[Tensor, Tuple[Tensor, Tensor]],
        edge_index: SparseTensor,
        edge_weight: Optional[Tensor] = None,
        size: Optional[Tuple[int, int]] = None
    ) -> Tensor:
        if isinstance(x, Tensor):
            x_src = x
        else:
            x_src, _ = x
        
        return edge_index.matmul(x_src)