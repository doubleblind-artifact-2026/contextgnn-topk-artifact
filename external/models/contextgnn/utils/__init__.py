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

from enum import Enum

import torch

class RHSEmbeddingMode(Enum):
    r"""Specifies how to incorporate shallow RHS representations in link prediction tasks."""

    LOOKUP = "lookup"
    FEATURE = "feature"
    FUSION = "fusion"

def sparse_matrix_to_sparse_coo(sci_sparse_matrix):
    r"""Converts scipy sparse matrix to sparse coo matrix."""

    sci_sparse_coo = sci_sparse_matrix.tocoo()

    values = torch.tensor(sci_sparse_coo.data, dtype=torch.int64)

    row_indices = torch.tensor(sci_sparse_coo.row, dtype=torch.int64)
    col_indices = torch.tensor(sci_sparse_coo.col, dtype=torch.int64)

    torch_sparse_tensor = torch.sparse_coo_tensor(
        indices=torch.stack([row_indices, col_indices]),
        values=values,
        size=sci_sparse_matrix.shape
    )

    return torch_sparse_tensor

__all__ = classes = [
    "RHSEmbeddingMode",
    "sparse_matrix_to_sparse_coo"
]