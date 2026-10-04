"""
This file is adapted from code originally released in the ContextGNN project.

Original source:
- https://github.com/kumo-ai/ContextGNN

The original code is distributed under the MIT License. The corresponding
copyright and license notices are preserved in this repository.

Modifications were made for the experiments accompanying this submission.
"""

from typing import List, Optional

import torch
from torch import Tensor
from torch_frame import TensorFrame
from torch_frame.nn import StypeWiseFeatureEncoder
from torch_frame.nn.models.resnet import FCResidualBlock
from typing_extensions import Self

from ..utils import RHSEmbeddingMode

class RHSEmbedding(torch.nn.Module):
    r"""RHSEmbedding module for GNNs."""

    def __init__(
        self,
        emb_mode: RHSEmbeddingMode,
        embedding_dim: int,
        num_nodes: int,
        col_stats: Optional[dict] = None,
        col_names_dict: Optional[dict] = None,
        stype_encoder_dict: Optional[dict] = None,
        feat: Optional[TensorFrame] = None,
        dense_feat: Optional[Tensor] = None
    ):
        super().__init__()

        self.emb_mode = emb_mode
        self.encoder: Optional[StypeWiseFeatureEncoder] = None
        self.dense_encoder: Optional[torch.nn.Linear] = None
        self.projector: Optional[torch.nn.Sequential] = None
        self._feat = feat
        self._dense_feat = dense_feat

        if self.emb_mode in [RHSEmbeddingMode.FEATURE, RHSEmbeddingMode.FUSION]:
            seqs: List[torch.nn.Module] = []

            if feat is not None:
                if col_stats is None or col_names_dict is None or stype_encoder_dict is None:
                    raise ValueError("TensorFrame backend requires col_stats, col_names_dict and stype_encoder_dict")

                self.encoder = StypeWiseFeatureEncoder(
                    out_channels=embedding_dim,
                    col_stats=col_stats,
                    col_names_dict=col_names_dict,
                    stype_encoder_dict=stype_encoder_dict
                )
            elif dense_feat is not None:
                self.dense_encoder = torch.nn.Linear(dense_feat.size(-1), embedding_dim)
            else:
                raise ValueError(f"RHSEmbedding mode {self.emb_mode} requires feat or dense_feat")

            seqs += [
                FCResidualBlock(embedding_dim, embedding_dim),
                FCResidualBlock(embedding_dim, embedding_dim),
                torch.nn.LayerNorm(embedding_dim, eps=1e-7)
            ]

            self.projector = torch.nn.Sequential(*seqs)

        self.lookup_embedding: Optional[torch.nn.Embedding] = None

        if self.emb_mode in [RHSEmbeddingMode.LOOKUP, RHSEmbeddingMode.FUSION]:
            self.lookup_embedding = torch.nn.Embedding(num_nodes, embedding_dim)
        
        self._cached_rhs_embedding: Optional[Tensor] = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.lookup_embedding is not None:
            self.lookup_embedding.reset_parameters()

        if self.encoder is not None:
            self.encoder.reset_parameters()
        
        if self.dense_encoder is not None:
            self.dense_encoder.reset_parameters()
            
        if self.projector is not None:
            for child in self.projector.children():
                child.reset_parameters()
        
        self._cached_rhs_embedding = None
    
    def train(self, mode: bool = True):
        if mode:
            self._cached_rhs_embedding = None
        
        return super().train(mode)

    def forward(self) -> Tensor:
        if not self.training:
            if self._cached_rhs_embedding is not None:
                return self._cached_rhs_embedding
        
        outs = []

        if self.lookup_embedding is not None:
            outs.append(self.lookup_embedding.weight)
        
        if self.encoder is not None and self.projector is not None:
            assert self._feat is not None

            out = self.encoder(self._feat)[0]
            out = self.projector(out)
            out = torch.sum(out, dim=1)

            outs.append(out)
        elif self.dense_encoder is not None and self.projector is not None:
            assert self._dense_feat is not None

            out = self.dense_encoder(self._dense_feat)
            out = self.projector(out)

            outs.append(out)
        
        result = sum(outs)

        assert isinstance(result, Tensor)

        if not self.training:
            self._cached_rhs_embedding = result
        
        return result

    def eval(self):
        self._cached_rhs_embedding = None
        return super().eval()

    def to(self, *args, **kwargs) -> Self:
        if self._feat is not None:
            self._feat = self._feat.to(*args, **kwargs)
        
        if self._dense_feat is not None:
            self._dense_feat = self._dense_feat.to(*args, **kwargs)
        
        return super().to(*args, **kwargs)

    def cpu(self) -> Self:
        if self._feat is not None:
            self._feat = self._feat.cpu()
        
        if self._dense_feat is not None:
            self._dense_feat = self._dense_feat.cpu()
        
        return super().cpu()

    def cuda(self, *args, **kwargs) -> Self:
        if self._feat is not None:
            self._feat = self._feat.cuda(*args, **kwargs)
        
        if self._dense_feat is not None:
            self._dense_feat = self._dense_feat.cuda(*args, **kwargs)
        
        return super().cuda(*args, **kwargs)