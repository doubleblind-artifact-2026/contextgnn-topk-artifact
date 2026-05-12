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

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from tqdm import tqdm
import torch
import os
import psutil
import random
import time
import math
import hashlib

from elliot.utils.write import store_recommendation
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from .ContextGNNModel import ContextGNNModel
from .utils import RHSEmbeddingMode
from typing import Any, Dict, Tuple
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_frame.data.stats import StatType
from relbench.modeling.graph import LinkTrainTableInput
from relbench.modeling.loader import SparseTensor
from torch_geometric.seed import seed_everything
from torch_geometric.utils import sort_edge_index
from relbench.modeling.utils import to_unix_time
from torch_frame.utils import infer_df_stype
from torch import Tensor
from torch_frame import stype
from torch_geometric.utils.cross_entropy import sparse_cross_entropy
from torch_frame.data import Dataset
from torch_geometric.typing import NodeType
from torch_sparse import SparseTensor as TSSparseTensor
import pandas as pd
import numpy as np
from ast import literal_eval as make_tuple

from contextlib import nullcontext

PSEUDO_TIME = "pseudo_time"
SRC_ENTITY_TABLE = "user_table"
DST_ENTITY_TABLE = "item_table"
TRANSACTION_TABLE = "transaction_table"
SRC_ENTITY_COL = "user_id"
DST_ENTITY_COL = "item_id"

def _parse_inference_for_ranking(val):
    if val is None:
        modes = ["full"]
    elif isinstance(val, (list, tuple)):
        modes = [str(x).strip().lower() for x in val]
    else:
        s = str(val).strip()

        if s.lower() == "all":
            modes = ["full", "global", "local"]
        else:
            s = s.strip("()[]")
            modes = [m.strip().lower() for m in s.split(",") if m.strip()]

    allowed = {"full", "global", "local"}

    if len(modes) == 0:
        modes = ["full"]

    for m in modes:
        if m not in allowed:
            raise ValueError(f"inference_for_ranking contains '{m}' not valid. Admitted: {sorted(allowed)}")

    out = []

    for m in modes:
        if m not in out:
            out.append(m)

    return out

def _parse_fusion_gate_version(val):
    version = str(val).strip().lower()
    allowed = {"v1", "v2", "v3"}

    if version not in allowed:
        raise ValueError(f"fusion_gate_version='{version}' not valid. Admitted: {sorted(allowed)}")

    return version

def _parse_bool_flag(val):
    if isinstance(val, bool):
        return val

    if isinstance(val, (int, float)):
        return bool(val)

    s = str(val).strip().lower()

    if s in {"true", "1", "yes", "y", "on"}:
        return True

    if s in {"false", "0", "no", "n", "off"}:
        return False

    raise ValueError(f"Cannot parse boolean flag from value='{val}'.")

class ContextGNN(RecMixin, BaseRecommenderModel):
    r"""
    ContextGNN: Beyond Two-Tower Recommendation Systems (https://arxiv.org/abs/2411.19513)
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        self._params_list = [
            ("_learning_rate", "lr", "lr", 0.0005, float, None),
            ("_factors", "factors", "factors", 64, int, None),
            ("_n_layers", "n_layers", "n_layers", 1, int, None),
            ("_norm", "norm", "norm", "layer_norm", str, None),
            ("_sup_rat", "sup_rat", "sup_rat", 0.5, float, None),
            ("_channels", "channels", "channels", 128, int, None),
            ("_max_steps", "max_steps", "max_steps", 2000, int, None),
            ("_gnn", "gnn", "gnn", "GraphSAGE", str, None),
            ("_mess_drop", "mess_drop", "mess_drop", 0.5, float, None),
            ("_eps", "eps", "eps", 1e-5, float, None),
            ("_t_eps", "t_eps", "t_eps", True, bool, None),
            ("_adj_norm", "adj_norm", "adj_norm", True, bool, None),
            ("_loss", "loss", "loss", "ce", str, None),
            ("_aggr", "aggr", "aggr", "sum", str, None),
            (
                "_neigh",
                "neigh",
                "neigh",
                "(16,16,16,16)",
                lambda x: list(make_tuple(str(x))),
                lambda x: self._batch_remove(str(x), " []").replace(",", "-")
            ),
            ("_acc_steps", "acc_steps", "acc_steps", 1, int, None),
            (
                "_inference_for_ranking",
                "inference_for_ranking",
                "ifr",
                "full",
                _parse_inference_for_ranking,
                lambda x: "-".join(x)
            ),
            (
                "_fusion_gate_version",
                "fusion_gate_version",
                "fgv",
                "v1",
                _parse_fusion_gate_version,
                lambda x: x
            ),
            ("_lambda_entropy", "lambda_entropy", "lambda_entropy", 1e-5, float, None),
            ("_gate_temperature", "gate_temperature", "gt", 1.0, float, None)
        ]

        self.autoset_params()

        self._fixed_gates = _parse_bool_flag(getattr(self._params, "fixed_gates", False))
        self._fixed_gate_values = (0.25, 0.50, 0.75)

        self._fixed_gates_weights_name = getattr(self._params, "fixed_gates_weights_name", None)

        if self._fixed_gates_weights_name is not None:
            self._fixed_gates_weights_name = str(self._fixed_gates_weights_name).strip()

            if len(self._fixed_gates_weights_name) == 0:
                self._fixed_gates_weights_name = None

        if self._fixed_gates and self._fixed_gates_weights_name is None:
            raise ValueError("fixed_gates=True requires 'fixed_gates_weights_name' in the config file.")

        if "full" in self._inference_for_ranking:
            self._primary_inference_mode = "full"
        else:
            self._primary_inference_mode = self._inference_for_ranking[0]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        seed_everything(self._seed)

        if "CUBLAS_WORKSPACE_CONFIG" in os.environ:
            del os.environ["CUBLAS_WORKSPACE_CONFIG"]

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.use_deterministic_algorithms(False)

        try:
            torch.set_float32_matmul_precision("medium")
        except Exception:
            pass
        
        if self._batch_size < 1:
            self._batch_size = self._num_users

        src_df, dst_df, target_df, train_df, val_df, test_df = data.get_relbench_df()
        del data

        NUM_SRC_NODES = len(src_df)
        self.NUM_DST_NODES = len(dst_df)

        table_dict = {
            SRC_ENTITY_TABLE: src_df,
            DST_ENTITY_TABLE: dst_df,
            TRANSACTION_TABLE: target_df
        }

        def get_static_stype_proposal(
            table_dict: Dict[str, pd.DataFrame]
        ) -> Dict[str, Dict[str, stype]]:
            inferred_col_to_stype_dict = {}

            for table_name, df in table_dict.items():
                df = df.sample(min(1_000, len(df)))
                inferred_col_to_stype_dict[table_name] = infer_df_stype(df)
            
            return inferred_col_to_stype_dict

        col_to_stype_dict = get_static_stype_proposal(table_dict)

        def static_data_make_pkey_fkey_graph(
            table_dict: Dict[str, pd.DataFrame],
            col_to_stype_dict: Dict[str, Dict[str, stype]]
        ) -> Tuple[HeteroData, Dict[str, Dict[str, Dict[StatType, Any]]]]:
            data = HeteroData()
            col_stats_dict = dict()

            src_col_to_stype = {"__const__": stype.numerical}
            src_df_const = pd.DataFrame({"__const__": np.ones(len(table_dict[SRC_ENTITY_TABLE]))})
            src_dataset = Dataset(df=src_df_const, col_to_stype=src_col_to_stype).materialize()

            data[SRC_ENTITY_TABLE].tf = src_dataset.tensor_frame
            data[SRC_ENTITY_TABLE].time = torch.from_numpy(to_unix_time(table_dict[SRC_ENTITY_TABLE][PSEUDO_TIME]))
            
            col_stats_dict[SRC_ENTITY_TABLE] = src_dataset.col_stats

            dst_col_to_stype = {"__const__": stype.numerical}
            dst_df_const = pd.DataFrame({"__const__": np.ones(len(table_dict[DST_ENTITY_TABLE]))})
            dst_dataset = Dataset(df=dst_df_const, col_to_stype=dst_col_to_stype).materialize()

            data[DST_ENTITY_TABLE].tf = dst_dataset.tensor_frame
            data[DST_ENTITY_TABLE].time = torch.from_numpy(to_unix_time(table_dict[DST_ENTITY_TABLE][PSEUDO_TIME]))
            
            col_stats_dict[DST_ENTITY_TABLE] = dst_dataset.col_stats
            
            fkey_index = torch.from_numpy(table_dict[TRANSACTION_TABLE][SRC_ENTITY_COL].astype(int).values)
            pkey_index = torch.from_numpy(table_dict[TRANSACTION_TABLE][DST_ENTITY_COL].astype(int).values)

            edge_index = torch.stack([fkey_index, pkey_index], dim=0)
            edge_type = (SRC_ENTITY_TABLE, SRC_ENTITY_COL, DST_ENTITY_TABLE)
            data[edge_type].edge_index = sort_edge_index(edge_index)

            reverse_edge_index = torch.stack([pkey_index, fkey_index], dim=0)
            reverse_edge_type = (DST_ENTITY_TABLE, DST_ENTITY_COL, SRC_ENTITY_TABLE)
            data[reverse_edge_type].edge_index = sort_edge_index(reverse_edge_index)

            data.validate()

            return data, col_stats_dict

        data_graph, col_stats_dict = static_data_make_pkey_fkey_graph(table_dict=table_dict, col_to_stype_dict=col_to_stype_dict)

        num_neighbors = self._neigh

        self._neg = self._loss in ["bpr", "bce"]

        def static_get_link_train_table_input(
            transaction_df: pd.DataFrame,
            num_dst_nodes: int
        ) -> Tuple[LinkTrainTableInput, Tensor]:
            df = transaction_df

            src_node_idx: Tensor = torch.from_numpy(df[SRC_ENTITY_COL].astype(int).values)
            exploded = df[DST_ENTITY_COL].explode().dropna()

            coo_indices = torch.from_numpy(np.stack([exploded.index.values, exploded.values.astype(int)]))

            sparse_coo = torch.sparse_coo_tensor(
                coo_indices,
                torch.ones(coo_indices.size(1), dtype=bool),
                (len(src_node_idx), num_dst_nodes)
            )

            dst_node_indices = sparse_coo.to_sparse_csr()
            time_t = torch.from_numpy(to_unix_time(df[PSEUDO_TIME]))

            return (
                LinkTrainTableInput(
                    src_nodes=(SRC_ENTITY_TABLE, src_node_idx),
                    dst_nodes=(DST_ENTITY_TABLE, dst_node_indices),
                    num_dst_nodes=num_dst_nodes,
                    src_time=time_t
                ),
                coo_indices
            )

        self.loader_dict: Dict[str, NeighborLoader] = {}
        self.dst_nodes_dict: Dict[str, Tuple[str, Tensor]] = {}
        num_dst_nodes_dict: Dict[str, int] = {}

        split_to_table = {
            "train": train_df,
            "validation": val_df,
            "test": test_df
        }

        self._eval_batch_size = getattr(self._params, "eval_batch_size", self._batch_size)

        for split, table in split_to_table.items():
            if split == "train":
                table_input, self.coo_idxs = static_get_link_train_table_input(table, num_dst_nodes=self.NUM_DST_NODES)
            else:
                table_input, _ = static_get_link_train_table_input(table, num_dst_nodes=self.NUM_DST_NODES)

            batch_size = self._batch_size if split == "train" else self._eval_batch_size

            self.dst_nodes_dict[split] = table_input.dst_nodes
            num_dst_nodes_dict[split] = table_input.num_dst_nodes

            self.loader_dict[split] = NeighborLoader(
                data_graph,
                num_neighbors=num_neighbors,
                time_attr="time",
                input_nodes=table_input.src_nodes,
                input_time=table_input.src_time,
                subgraph_type="bidirectional",
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=4,
                persistent_workers=True,
                pin_memory=torch.cuda.is_available(),
                disjoint=True
            )

        self._model = ContextGNNModel(
            data=data_graph,
            col_stats_dict=col_stats_dict,
            rhs_emb_mode=RHSEmbeddingMode.FUSION,
            dst_entity_table=DST_ENTITY_TABLE,
            num_nodes=num_dst_nodes_dict["train"],
            num_layers=self._n_layers,
            channels=self._channels,
            aggr=self._aggr,
            normalize=self._adj_norm,
            dropout=self._mess_drop,
            norm=self._norm,
            lr=self._learning_rate,
            gnn=self._gnn,
            eps=self._eps,
            train_eps=self._t_eps,
            embedding_dim=self._factors,
            torch_frame_model_kwargs={"channels": self._channels, "num_layers": self._n_layers},
            is_static=True,
            src_entity_table=SRC_ENTITY_TABLE,
            num_src_nodes=NUM_SRC_NODES,
            fusion_gate_version=self._fusion_gate_version,
            gate_temperature=self._gate_temperature
        )

        self._save_resume = getattr(self._params.meta, "save_resume", True)

        self._resume_saving_filepath = os.path.abspath(
            os.sep.join([
                self._config.path_output_rec_weight,
                self.name,
                f"resume-state-{self.name}.pt"
            ])
        )

        self._start_epoch = 0

        self._final_test_results = None
        self._final_test_results_by_mode = None
        self._final_test_recommendations = None

    def _build_run_signature(self) -> str:
        es = getattr(self._params, "early_stopping", None)

        es_monitor = getattr(es, "monitor", "none") if es is not None else "none"
        es_patience = getattr(es, "patience", "none") if es is not None else "none"
        es_min_delta = getattr(es, "min_delta", "none") if es is not None else "none"
        es_mode = getattr(es, "mode", "none") if es is not None else "none"

        eval_batch_size = getattr(self._params, "eval_batch_size", self._batch_size)

        parts = [
            f"dataset={self._config.dataset}",
            self.get_base_params_shortcut(),
            self.get_params_shortcut(),
            f"eval_bs={eval_batch_size}",
            f"vr={self._validation_rate}",
            f"es_monitor={es_monitor}",
            f"es_patience={es_patience}",
            f"es_min_delta={es_min_delta}",
            f"es_mode={es_mode}"
        ]

        return "|".join(map(str, parts))

    @property
    def name(self):
        signature = self._build_run_signature()
        run_hash = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]

        return (
            f"ContextGNN_"
            f"{self._config.dataset}_"
            f"seed={self._seed}_"
            f"fgv={self._fusion_gate_version}_"
            f"h={run_hash}"
        )

    def _get_fixed_gates_best_weights_filepath(self) -> str:
        if self._fixed_gates_weights_name is None:
            raise ValueError(
                "Cannot build fixed-gates best checkpoint filepath: "
                "'fixed_gates_weights_name' is missing."
            )

        weights_name = self._fixed_gates_weights_name

        return os.path.abspath(
            os.path.join(
                ".",
                "results",
                "weights",
                weights_name
            )
        )

    def negative_sampling(self, edge_label_index):
        users = edge_label_index[0]
        positives = edge_label_index[1]

        negs = torch.randint(
            low=0,
            high=self._num_items,
            size=positives.shape,
            device=self.device
        )

        mask = negs == positives

        while mask.any():
            negs[mask] = torch.randint(
                low=0,
                high=self._num_items,
                size=(mask.sum(),),
                device=self.device
            )

            mask = negs == positives

        return torch.stack([users, negs], dim=0)

    def _mask_seen_items_inplace(self, preds: torch.Tensor, user_idx_np: np.ndarray, split_name: str) -> bool:
        split_name = str(split_name).lower().strip()

        if split_name == "validation":
            cache_attr = "_sp_seen_validation_csr"
            sp = getattr(self._data, "sp_i_train", None)
        elif split_name == "test":
            cache_attr = "_sp_seen_test_csr"
            sp = getattr(self._data, "sp_i_train_val", getattr(self._data, "sp_i_train", None))
        else:
            raise ValueError(f"Unknown split_name='{split_name}'")

        if sp is None:
            return False

        if not hasattr(self, cache_attr):
            if sp.shape[0] != self._num_users and sp.shape[1] == self._num_users:
                sp = sp.T
            
            setattr(self, cache_attr, sp.tocsr())

        sp = getattr(self, cache_attr)
        sp_batch = sp[user_idx_np]

        cols = sp_batch.indices

        if cols.size == 0:
            return True

        indptr = sp_batch.indptr
        nnz_per_row = np.diff(indptr)
        rows = np.repeat(np.arange(len(user_idx_np), dtype=np.int64), nnz_per_row)

        row_t = torch.from_numpy(rows).to(preds.device, non_blocking=True)
        col_t = torch.from_numpy(cols.astype(np.int64)).to(preds.device, non_blocking=True)
        preds[row_t, col_t] = float("-inf")

        return True

    def _is_sparse_gnn(self) -> bool:
        return str(self._gnn).lower() in {"lightgcn", "ngcf"}

    def _edge_index_to_torch_sparse_adj_t(
        self,
        edge_index_2xe: torch.Tensor,
        num_src: int,
        num_dst: int,
        device: torch.device
    ) -> TSSparseTensor:
        src = edge_index_2xe[0]
        dst = edge_index_2xe[1]
        val = torch.ones(src.numel(), device=device, dtype=torch.float32)

        return TSSparseTensor(
            row=dst,
            col=src,
            value=val,
            sparse_sizes=(num_dst, num_src)
        ).coalesce()

    def _maybe_convert_eval_edges_to_torch_sparse(self, batch: HeteroData) -> None:
        if not self._is_sparse_gnn():
            return

        edge_type = (SRC_ENTITY_TABLE, SRC_ENTITY_COL, DST_ENTITY_TABLE)
        rev_edge_type = (DST_ENTITY_TABLE, DST_ENTITY_COL, SRC_ENTITY_TABLE)

        num_src = batch.num_nodes_dict[SRC_ENTITY_TABLE]
        num_dst = batch.num_nodes_dict[DST_ENTITY_TABLE]

        ei = batch[edge_type].edge_index

        if not isinstance(ei, TSSparseTensor):
            batch[edge_type].edge_index = self._edge_index_to_torch_sparse_adj_t(ei, num_src=num_src, num_dst=num_dst, device=self.device)
        
        ei_rev = batch[rev_edge_type].edge_index

        if not isinstance(ei_rev, TSSparseTensor):
            batch[rev_edge_type].edge_index = self._edge_index_to_torch_sparse_adj_t(ei_rev, num_src=num_dst, num_dst=num_src, device=self.device)
    
    def run_fixed_gate_tests(self):
        rng = self._rng_snapshot()

        try:
            k_need = self.evaluator.get_needed_recommendations()

            jaccard_k = 20
            modes = getattr(self, "_inference_for_ranking", ["full"])
            primary = getattr(self, "_primary_inference_mode", "full")

            self.logger.info("*** FIXED GATES TEST START ***")

            for g in self._fixed_gate_values:
                results_by_mode = {}
                recs_by_mode = {}

                for mode in modes:
                    test_recs = self._collect_recommendations_for_split(
                        "test",
                        k_need,
                        inference_mode=mode,
                        count_usage=True,
                        fixed_gate_value=(g if mode == "full" else None)
                    )

                    recs_by_mode[mode] = test_recs
                    results_by_mode[mode] = self.evaluator.eval_test(test_recs)

                    if getattr(self, "_usage_total", 0) > 0:
                        local_share = self._usage_local_hits / self._usage_total
                        global_share = 1.0 - local_share

                        self.logger.info(
                            f"[BRANCH USAGE@{k_need}, fixed_g={g:.2f}, mode={mode}] "
                            f"local_topk_share={local_share:.4f}; global_topk_share={global_share:.4f}"
                        )

                    if mode == "full":
                        self._log_gate_topk_stats(prefix=f"[GATE TOPK@20, fixed_g={g:.2f}, mode={mode}]")

                jaccard_full_local = None
                jaccard_full_global = None

                if "full" in recs_by_mode and "local" in recs_by_mode:
                    jaccard_full_local = self._mean_jaccard_at_k(
                        recs_by_mode["full"],
                        recs_by_mode["local"],
                        k=jaccard_k
                    )

                if "full" in recs_by_mode and "global" in recs_by_mode:
                    jaccard_full_global = self._mean_jaccard_at_k(
                        recs_by_mode["full"],
                        recs_by_mode["global"],
                        k=jaccard_k
                    )

                jaccard_parts = []

                if jaccard_full_local is not None:
                    jaccard_parts.append(f"Jaccard@{jaccard_k}(Full, Local)={jaccard_full_local:.4f}")

                if jaccard_full_global is not None:
                    jaccard_parts.append(f"Jaccard@{jaccard_k}(Full, Global)={jaccard_full_global:.4f}")

                if jaccard_parts:
                    self.logger.info(f"[RANKING OVERLAP, fixed_g={g:.2f}] " + "; ".join(jaccard_parts))

                primary_result = results_by_mode.get(primary, None)
                primary_recs = recs_by_mode.get(primary, None)

                if primary_result is not None:
                    self.logger.info(
                        f"[FIXED GATE SUMMARY, g={g:.2f}, primary_mode={primary}] "
                        f"{primary_result[self._validation_k]['test_results']}"
                    )

                if self._save_recs and primary_recs is not None:
                    gate_tag = f"{g:.2f}".replace(".", "p")

                    self.logger.info(f"Writing recommendations at: {self._config.path_output_rec_result}")

                    store_recommendation(
                        primary_recs,
                        os.path.abspath(
                            os.sep.join([
                                self._config.path_output_rec_result,
                                f"{self.name}_fixedg={gate_tag}_mode={primary}.tsv"
                            ])
                        )
                    )

            self.logger.info("*** FIXED GATES TEST END ***")
        finally:
            self._rng_restore(rng)

    def train(self):
        restored = False
        self._start_epoch = 0

        if self._fixed_gates:
            fixed_gates_best_filepath = self._get_fixed_gates_best_weights_filepath()

            self.logger.info("fixed_gates=True detected: skipping training and running fixed-gate test on explicitly selected best weights only.")
            self.logger.info(f"Fixed-gates best checkpoint path: '{fixed_gates_best_filepath}'")

            restored_best = self.restore_best_weights(filepath=fixed_gates_best_filepath)

            if not restored_best:
                raise FileNotFoundError(f"Fixed-gates test requested, but no best checkpoint was found at '{fixed_gates_best_filepath}'.")

            self.run_fixed_gate_tests()
        
            return

        if self._restore:
            restored = self.restore_training_state()

            if restored:
                self.logger.info("Starting training from restored resume checkpoint.")
            else:
                self.logger.info("No valid resume checkpoint restored. Starting training from scratch.")

        if self._start_epoch >= self._epochs:
            self.logger.info(
                f"Training already completed: completed_epochs={self._start_epoch}, "
                f"total_epochs={self._epochs}."
            )

            best_restored = self.restore_best_weights()

            if not best_restored:
                self.logger.warning("Best checkpoint not found. Final test will use the current in-memory weights.")

            self._run_final_test()
            
            return

        process = psutil.Process()

        for it in range(self._start_epoch, self._epochs):
            if self._early_stopping.stop(self._losses[:], self._results):
                self.logger.info(f"Met Early Stopping conditions: {self._early_stopping}")
                break

            epoch_start = time.perf_counter()
            self._model.train()

            loss_accum = 0.0
            loss_sq_accum = 0.0
            count_accum = 0
            batch_loss_min = float("inf")
            batch_loss_max = 0.0
            steps = 0

            grad_norm_sum = 0.0
            grad_norm_max = 0.0
            grad_norm_steps = 0

            supervised_edges_sum = 0
            supervised_edges_min = float("inf")
            supervised_edges_max = 0

            max_cpu_mem = 0.0
            max_gpu_mem = 0.0

            metric_acc = {}

            total_steps = min(len(self.loader_dict["train"]), self._max_steps)
            sparse_tensor = SparseTensor(self.dst_nodes_dict["train"][1], device=self.device)

            self._model.optimizer.zero_grad()

            with tqdm(total=total_steps, disable=not self._verbose) as t:
                for i, batch in enumerate(self.loader_dict["train"]):
                    batch = batch.to(self.device, non_blocking=True)

                    for nt in batch.node_types:
                        if hasattr(batch[nt], "n_id"):
                            batch[nt].n_id = batch[nt].n_id.to(self.device, non_blocking=True)

                        if hasattr(batch[nt], "batch"):
                            batch[nt].batch = batch[nt].batch.to(self.device, non_blocking=True)

                    for key, val in batch.n_id_dict.items():
                        batch.n_id_dict[key] = val.to(self.device, non_blocking=True)

                    for key, val in batch.batch_dict.items():
                        batch.batch_dict[key] = val.to(self.device, non_blocking=True)

                    input_id = batch[SRC_ENTITY_TABLE].input_id
                    src_batch, dst_index = sparse_tensor[input_id]
                    edge_label_index = torch.stack([src_batch, dst_index], dim=0)

                    train_seed_nodes = self.loader_dict["train"].input_nodes[1].to(self.device)
                    global_src_index = train_seed_nodes[batch[SRC_ENTITY_TABLE].input_id[src_batch]]
                    global_edge_label_index = torch.stack([global_src_index, dst_index], dim=0)

                    supervision_edges_sample_size = int(global_edge_label_index.shape[1] * self._sup_rat)

                    sample_indices = torch.randperm(
                        global_edge_label_index.shape[1],
                        device=global_edge_label_index.device
                    )[:supervision_edges_sample_size]

                    global_edge_label_index_sample = global_edge_label_index[:, sample_indices]
                    edge_label_index = edge_label_index[:, sample_indices]

                    num_supervised = global_edge_label_index_sample.shape[1]
                    supervised_edges_sum += num_supervised
                    supervised_edges_min = min(supervised_edges_min, num_supervised)
                    supervised_edges_max = max(supervised_edges_max, num_supervised)

                    if self._neg:
                        edge_label_index_neg = self.negative_sampling(global_edge_label_index_sample)
                        edge_label_index_neg[0] = edge_label_index[0]

                    edge_type = (SRC_ENTITY_TABLE, SRC_ENTITY_COL, DST_ENTITY_TABLE)
                    edge_index = batch[edge_type].edge_index

                    global_src_index = batch[SRC_ENTITY_TABLE].n_id[edge_index[0]]
                    global_dst_index = batch[DST_ENTITY_TABLE].n_id[edge_index[1]]
                    global_edge_index = torch.stack([global_src_index, global_dst_index])

                    global_src_batch = batch[SRC_ENTITY_TABLE].batch[edge_index[0]]
                    global_seed_nodes = train_seed_nodes[batch[SRC_ENTITY_TABLE].input_id[global_src_batch]]
                    supervision_seed_node_mask = (global_seed_nodes == global_edge_index[0])

                    global_edge_label_index_hash = (global_edge_label_index_sample[0, :] * self.NUM_DST_NODES + global_edge_label_index_sample[1, :])
                    global_edge_index_hash = (global_edge_index[0, :] * self.NUM_DST_NODES + global_edge_index[1, :])

                    mask = ~(torch.isin(global_edge_index_hash, global_edge_label_index_hash) * supervision_seed_node_mask)
                    edge_index_message_passing = edge_index[:, mask]

                    values = torch.ones(edge_index_message_passing.size(1), device=self.device)

                    edge_type = (SRC_ENTITY_TABLE, SRC_ENTITY_COL, DST_ENTITY_TABLE)
                    reverse_edge_type = (DST_ENTITY_TABLE, DST_ENTITY_COL, SRC_ENTITY_TABLE)

                    num_src = batch.num_nodes_dict[SRC_ENTITY_TABLE]
                    num_dst = batch.num_nodes_dict[DST_ENTITY_TABLE]

                    src = edge_index_message_passing[0]
                    dst = edge_index_message_passing[1]
                    values_f32 = values.to(torch.float32)

                    if self._is_sparse_gnn():
                        adj_t_dst_src = TSSparseTensor(row=dst, col=src, value=values_f32, sparse_sizes=(num_dst, num_src)).coalesce()
                        adj_t_src_dst = TSSparseTensor(row=src, col=dst, value=values_f32, sparse_sizes=(num_src, num_dst)).coalesce()

                        batch[edge_type].edge_index = adj_t_dst_src
                        batch[reverse_edge_type].edge_index = adj_t_src_dst
                    else:
                        edge_index_message_passing_sparse = torch.sparse_coo_tensor(
                            edge_index_message_passing,
                            values,
                            size=(num_src, num_dst)
                        )

                        edge_index_message_passing_reverse_sparse = torch.sparse_coo_tensor(
                            edge_index_message_passing.flip(0),
                            values,
                            size=(num_dst, num_src)
                        )

                        batch[edge_type].edge_index = edge_index_message_passing_reverse_sparse
                        batch[reverse_edge_type].edge_index = edge_index_message_passing_sparse

                    logits = self._model(batch, SRC_ENTITY_TABLE, DST_ENTITY_TABLE)

                    if self._loss == "ce":
                        loss = sparse_cross_entropy(logits, edge_label_index)
                    elif self._loss == "bpr":
                        pos = logits[edge_label_index[0], edge_label_index[1]]
                        neg = logits[edge_label_index_neg[0], edge_label_index_neg[1]]
                        
                        loss = torch.mean(torch.nn.functional.softplus(neg - pos))
                    elif self._loss == "bce":
                        pos_logits = logits[edge_label_index[0], edge_label_index[1]]
                        neg_logits = logits[edge_label_index_neg[0], edge_label_index_neg[1]]
                        
                        logits_all = torch.cat([pos_logits, neg_logits], dim=0)
                        labels_all = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)
                        
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits_all, labels_all)
                    else:
                        raise NotImplementedError

                    g = getattr(self._model, "_last_gate_tensor", None)

                    if g is not None:
                        eps = 1e-8
                        entropy = -(g * torch.log(g + eps) + (1.0 - g) * torch.log(1.0 - g + eps))
                        loss = loss - self._lambda_entropy * entropy.mean()

                    (loss / self._acc_steps).backward()

                    batch_loss_val = float(loss)
                    loss_weight = max(int(num_supervised), 1)

                    loss_accum += batch_loss_val * loss_weight
                    loss_sq_accum += (batch_loss_val ** 2) * loss_weight
                    count_accum += loss_weight

                    batch_loss_min = min(batch_loss_min, batch_loss_val)
                    batch_loss_max = max(batch_loss_max, batch_loss_val)

                    monitor = getattr(self._model, "_monitor", None)

                    if monitor is not None:
                        user_weight = int(batch[SRC_ENTITY_TABLE].seed_time.size(0))
                        pair_weight = user_weight * self.NUM_DST_NODES
                        local_item_weight = max(int(monitor.get("num_overrides", 0)), 1)
                        rhs_gnn_weight = int(batch[DST_ENTITY_TABLE].n_id.numel())
                        rhs_emb_weight = self.NUM_DST_NODES

                        self._weighted_add(metric_acc, "emb_logit_mean", monitor.get("emb_logit_mean", 0.0), pair_weight)
                        self._weighted_add(metric_acc, "emb_logit_std", monitor.get("emb_logit_std", 0.0), pair_weight)

                        self._weighted_add(metric_acc, "emb_logit_pre_mean", monitor.get("emb_logit_pre_mean", 0.0), pair_weight)
                        self._weighted_add(metric_acc, "emb_logit_pre_std", monitor.get("emb_logit_pre_std", 0.0), pair_weight)

                        self._weighted_add(metric_acc, "id_logit_mean", monitor.get("id_logit_mean", 0.0), local_item_weight)
                        self._weighted_add(metric_acc, "id_logit_std", monitor.get("id_logit_std", 0.0), local_item_weight)

                        self._weighted_add(metric_acc, "lhs_norm_mean", monitor.get("lhs_norm_mean", 0.0), user_weight)
                        self._weighted_add(metric_acc, "lhs_norm_std", monitor.get("lhs_norm_std", 0.0), user_weight)
                        self._weighted_add(metric_acc, "lhs_dispersion", monitor.get("lhs_dispersion", 0.0), user_weight)

                        self._weighted_add(metric_acc, "rhs_gnn_norm_mean", monitor.get("rhs_gnn_norm_mean", 0.0), rhs_gnn_weight)
                        self._weighted_add(metric_acc, "rhs_gnn_norm_std", monitor.get("rhs_gnn_norm_std", 0.0), rhs_gnn_weight)
                        self._weighted_add(metric_acc, "rhs_gnn_dispersion", monitor.get("rhs_gnn_dispersion", 0.0), rhs_gnn_weight)

                        self._weighted_add(metric_acc, "rhs_emb_norm_mean", monitor.get("rhs_emb_norm_mean", 0.0), rhs_emb_weight)
                        self._weighted_add(metric_acc, "rhs_emb_norm_std", monitor.get("rhs_emb_norm_std", 0.0), rhs_emb_weight)
                        self._weighted_add(metric_acc, "rhs_emb_dispersion", monitor.get("rhs_emb_dispersion", 0.0), rhs_emb_weight)

                        self._weighted_add(metric_acc, "gate_mean", monitor.get("gate_mean", 0.0), local_item_weight)
                        self._weighted_add(metric_acc, "gate_std", monitor.get("gate_std", 0.0), local_item_weight)
                        self._weighted_add(metric_acc, "gate_min", monitor.get("gate_min", 0.0), local_item_weight)
                        self._weighted_add(metric_acc, "gate_max", monitor.get("gate_max", 0.0), local_item_weight)
                        self._weighted_add(metric_acc, "gate_entropy_mean", monitor.get("gate_entropy_mean", 0.0), local_item_weight)

                        if "gate_temperature" in monitor:
                            self._weighted_add(metric_acc, "gate_temperature", monitor["gate_temperature"], local_item_weight)

                        self._weighted_add(metric_acc, "override_ratio", monitor.get("override_ratio", 0.0), pair_weight)
                        self._weighted_add(metric_acc, "local_items_per_user_mean", monitor.get("local_items_per_user_mean", 0.0), user_weight)
                        self._weighted_add(metric_acc, "local_items_per_user_std", monitor.get("local_items_per_user_std", 0.0), user_weight)
                        self._weighted_add(metric_acc, "delta_override_mean", monitor.get("delta_override_mean", 0.0), local_item_weight)
                        self._weighted_add(metric_acc, "delta_override_std", monitor.get("delta_override_std", 0.0), local_item_weight)

                    if ((i + 1) % self._acc_steps == 0) or ((i + 1) == len(self.loader_dict["train"])):
                        total_norm_sq = 0.0

                        for p in self._model.parameters():
                            if p.grad is not None:
                                param_norm = p.grad.data.norm(2)
                                total_norm_sq += param_norm.item() ** 2

                        total_norm = math.sqrt(total_norm_sq) if total_norm_sq > 0.0 else 0.0

                        grad_norm_sum += total_norm
                        grad_norm_max = max(grad_norm_max, total_norm)
                        grad_norm_steps += 1

                        self._model.optimizer.step()
                        self._model.optimizer.zero_grad()

                    steps += 1

                    if steps >= self._max_steps:
                        break
                    
                    mem_usage = process.memory_info().rss / (1024**3)
                    max_cpu_mem = max(max_cpu_mem, mem_usage)

                    if torch.cuda.is_available():
                        gpu_usage = torch.cuda.max_memory_reserved(self.device) / (1024**3)
                        max_gpu_mem = max(max_gpu_mem, gpu_usage)
                    else:
                        gpu_usage = 0.0

                    current_epoch_loss = loss_accum / max(count_accum, 1)

                    t.set_postfix({"loss": f"{current_epoch_loss:.5f}", "cpu": f"{mem_usage:.2f}GB", "gpu": f"{gpu_usage:.2f}GB"})
                    t.update()

            epoch_loss = loss_accum / count_accum if count_accum > 0 else float("nan")
            mean_sq_loss = loss_sq_accum / count_accum if count_accum > 0 else float("nan")
            epoch_loss_std = (math.sqrt(max(mean_sq_loss - epoch_loss**2, 0.0)) if count_accum > 0 else float("nan"))

            emb_logit_mean_avg = self._weighted_mean(metric_acc, "emb_logit_mean")
            emb_logit_std_avg = self._weighted_mean(metric_acc, "emb_logit_std")
            id_logit_mean_avg = self._weighted_mean(metric_acc, "id_logit_mean")
            id_logit_std_avg = self._weighted_mean(metric_acc, "id_logit_std")

            lhs_norm_mean_avg = self._weighted_mean(metric_acc, "lhs_norm_mean")
            lhs_norm_std_avg = self._weighted_mean(metric_acc, "lhs_norm_std")
            rhs_gnn_norm_mean_avg = self._weighted_mean(metric_acc, "rhs_gnn_norm_mean")
            rhs_gnn_norm_std_avg = self._weighted_mean(metric_acc, "rhs_gnn_norm_std")
            rhs_emb_norm_mean_avg = self._weighted_mean(metric_acc, "rhs_emb_norm_mean")
            rhs_emb_norm_std_avg = self._weighted_mean(metric_acc, "rhs_emb_norm_std")

            lhs_disp_avg = self._weighted_mean(metric_acc, "lhs_dispersion")
            rhs_gnn_disp_avg = self._weighted_mean(metric_acc, "rhs_gnn_dispersion")
            rhs_emb_disp_avg = self._weighted_mean(metric_acc, "rhs_emb_dispersion")

            gate_mean_avg = self._weighted_mean(metric_acc, "gate_mean")
            gate_std_avg = self._weighted_mean(metric_acc, "gate_std")
            gate_min_avg = self._weighted_mean(metric_acc, "gate_min")
            gate_max_avg = self._weighted_mean(metric_acc, "gate_max")
            gate_entropy_mean_avg = self._weighted_mean(metric_acc, "gate_entropy_mean")
            gate_temperature_avg = self._weighted_mean(metric_acc, "gate_temperature")

            emb_logit_pre_mean_avg = self._weighted_mean(metric_acc, "emb_logit_pre_mean")
            emb_logit_pre_std_avg = self._weighted_mean(metric_acc, "emb_logit_pre_std")
            override_ratio_avg = self._weighted_mean(metric_acc, "override_ratio")
            local_items_per_user_mean_avg = self._weighted_mean(metric_acc, "local_items_per_user_mean")
            local_items_per_user_std_avg = self._weighted_mean(metric_acc, "local_items_per_user_std")
            delta_override_mean_avg = self._weighted_mean(metric_acc, "delta_override_mean")
            delta_override_std_avg = self._weighted_mean(metric_acc, "delta_override_std")

            grad_norm_mean = grad_norm_sum / grad_norm_steps if grad_norm_steps > 0 else float("nan")
            avg_supervised_edges = supervised_edges_sum / max(steps, 1) if supervised_edges_sum > 0 else 0.0

            epoch_time = time.perf_counter() - epoch_start
            avg_batch_time = epoch_time / max(steps, 1)

            gate_temperature_part = ""

            if self._fusion_gate_version == "v3":
                gate_temperature_part = f"gate_temperature={gate_temperature_avg:.4f}; "

            self.logger.info(
                f"[EPOCH {it + 1}/{self._epochs}] "
                f"train_loss={epoch_loss:.5f}; "
                f"train_loss_min={batch_loss_min:.5f}; train_loss_max={batch_loss_max:.5f}; train_loss_std={epoch_loss_std:.5f}; "
                f"grad_norm_mean={grad_norm_mean:.4f}; grad_norm_max={grad_norm_max:.4f}; "
                f"emb_logits_mean={emb_logit_mean_avg:.4f}; emb_logits_std={emb_logit_std_avg:.4f}; "
                f"emb_logits_pre_mean={emb_logit_pre_mean_avg:.4f}; emb_logits_pre_std={emb_logit_pre_std_avg:.4f}; "
                f"override_ratio={override_ratio_avg:.6f}; "
                f"local_items_per_user_mean={local_items_per_user_mean_avg:.2f}; local_items_per_user_std={local_items_per_user_std_avg:.2f}; "
                f"delta_override_mean={delta_override_mean_avg:.4f}; delta_override_std={delta_override_std_avg:.4f}; "
                f"gate_mean={gate_mean_avg:.4f}; gate_std={gate_std_avg:.4f}; "
                f"gate_min={gate_min_avg:.4f}; gate_max={gate_max_avg:.4f}; "
                f"gate_entropy_mean={gate_entropy_mean_avg:.4f}; "
                f"{gate_temperature_part}"
                f"id_logits_mean={id_logit_mean_avg:.4f}; id_logits_std={id_logit_std_avg:.4f}; "
                f"user_norm_mean={lhs_norm_mean_avg:.4f}; user_norm_std={lhs_norm_std_avg:.4f}; "
                f"user_dispersion={lhs_disp_avg:.4e}; "
                f"item_loc_norm_mean={rhs_gnn_norm_mean_avg:.4f}; item_loc_norm_std={rhs_gnn_norm_std_avg:.4f}; "
                f"item_loc_dispersion={rhs_gnn_disp_avg:.4e}; "
                f"rhs_emb_norm_mean={rhs_emb_norm_mean_avg:.4f}; rhs_emb_norm_std={rhs_emb_norm_std_avg:.4f}; "
                f"rhs_emb_dispersion={rhs_emb_disp_avg:.4e}; "
                f"supervised_edges_avg={avg_supervised_edges:.1f}; "
                f"supervised_edges_min={supervised_edges_min}; supervised_edges_max={supervised_edges_max}; "
                f"cpu_mem_max={max_cpu_mem:.2f}GB, gpu_mem_max={max_gpu_mem:.2f}GB; "
                f"epoch_time={epoch_time:.2f}s; batch_time_avg={avg_batch_time:.4f}s"
            )

            self.evaluate(it, epoch_loss)
            self._save_resume_checkpoint(last_completed_epoch=it)
        
        best_restored = self.restore_best_weights()

        if not best_restored:
            self.logger.warning("Best checkpoint not found. Final test will use the current in-memory weights.")

        self._run_final_test()

    def _reset_usage_trackers(self):
        self._usage_local_hits = 0
        self._usage_total = 0

    def _reset_gate_topk_trackers(self):
        self._topk_gate_sum_at20 = 0.0
        self._topk_gate_count_at20 = 0
        self._topk_local_hits_at20 = 0
        self._topk_local_high_gate_hits_at20 = 0

    def _log_gate_topk_stats(self, prefix: str):
        if (getattr(self, "_topk_local_hits_at20", 0) > 0) or (getattr(self, "_topk_gate_count_at20", 0) > 0):
            if self._topk_gate_count_at20 > 0:
                topk_local_gate_mean_20 = self._topk_gate_sum_at20 / self._topk_gate_count_at20
            else:
                topk_local_gate_mean_20 = float("nan")

            if self._topk_local_hits_at20 > 0:
                topk_local_with_gate_high_share_20 = self._topk_local_high_gate_hits_at20 / self._topk_local_hits_at20
            else:
                topk_local_with_gate_high_share_20 = float("nan")

            self.logger.info(
                f"{prefix} "
                f"topk_local_gate_mean@20={topk_local_gate_mean_20:.4f}; "
                f"topk_local_with_gate_high_share@20={topk_local_with_gate_high_share_20:.4f}"
            )

    def _collect_recommendations_for_split(
        self,
        split_name: str,
        k: int = 100,
        inference_mode: str = "full",
        count_usage: bool = False,
        fixed_gate_value: float = None
    ):
        predictions_top_k = {}

        self._reset_usage_trackers()

        if count_usage:
            self._reset_gate_topk_trackers()

        self._model.eval()

        use_cuda = torch.cuda.is_available()
        gnn_name = str(self._gnn).lower()
        sparse_gnn = gnn_name in {"lightgcn", "ngcf"}

        use_bf16 = (not sparse_gnn) and use_cuda and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        use_fp16 = use_cuda and (sparse_gnn or not use_bf16)

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else
            (torch.autocast(device_type="cuda", dtype=torch.float16) if use_fp16 else nullcontext())
        )

        with torch.no_grad():
            with amp_ctx:
                for batch in self.loader_dict[split_name]:
                    batch = batch.to(self.device, non_blocking=True)
                    self._maybe_convert_eval_edges_to_torch_sparse(batch)

                    batch_size = batch[SRC_ENTITY_TABLE].batch_size

                    scores = self._model(
                        batch,
                        SRC_ENTITY_TABLE,
                        DST_ENTITY_TABLE,
                        score_mode=inference_mode,
                        fixed_gate_value=fixed_gate_value
                    )

                    global_batch_src_ids = batch[SRC_ENTITY_TABLE].n_id[:batch_size]

                    mask = self.get_candidate_mask(validation=(split_name == "validation"))

                    recs = self.get_single_recommendation(
                        mask,
                        k,
                        scores,
                        global_batch_src_ids,
                        count_usage=count_usage,
                        split_name=split_name
                    )

                    predictions_top_k.update(recs)

        return predictions_top_k

    def get_recommendations(self, k: int = 100, inference_mode: str = "full", fixed_gate_value: float = None):
        self._reset_usage_trackers()
        self._reset_gate_topk_trackers()
        
        predictions_top_k_val = self._collect_recommendations_for_split(
            "validation",
            k,
            inference_mode="full",
            count_usage=False,
            fixed_gate_value=None
        )

        predictions_top_k_test = self._collect_recommendations_for_split(
            "test",
            k,
            inference_mode=inference_mode,
            count_usage=True,
            fixed_gate_value=fixed_gate_value
        )

        return predictions_top_k_val, predictions_top_k_test

    def get_single_recommendation(
        self,
        mask,
        k,
        predictions,
        global_batch_src_ids,
        count_usage: bool = True,
        split_name: str = "test"
    ):
        idx = global_batch_src_ids.detach().cpu().numpy()

        preds = predictions

        if not torch.is_tensor(preds):
            preds = torch.as_tensor(preds)
        
        preds = preds.to(self.device, non_blocking=True)

        if not self._negative_sampling:
            applied = self._mask_seen_items_inplace(preds, idx, split_name=split_name)

            if applied:
                v, i = torch.topk(preds, k=k, dim=1, sorted=True)
            else:
                filtered_mask = mask[idx]
                v, i = self._model.get_top_k(preds, filtered_mask, k=k)
        else:
            filtered_mask = mask[idx]
            v, i = self._model.get_top_k(preds, filtered_mask, k=k)
        
        if count_usage:
            local_assign = getattr(self._model, "_last_local_assign", None)
            gate_assign = getattr(self._model, "_last_gate_assign", None)

            if local_assign is not None:
                lhs_b, rhs_idx, bsz = local_assign

                if bsz == i.size(0):
                    local_sets = [set() for _ in range(bsz)]

                    for u_pos, item in zip(lhs_b.tolist(), rhs_idx.tolist()):
                        local_sets[u_pos].add(int(item))

                    gate_maps = None

                    if gate_assign is not None:
                        lhs_bg, rhs_idxg, gate_vals, bsz_g = gate_assign

                        if bsz_g == i.size(0):
                            gate_maps = [dict() for _ in range(bsz_g)]

                            for u_pos, item, gate_val in zip(lhs_bg.tolist(), rhs_idxg.tolist(), gate_vals.tolist()):
                                gate_maps[u_pos][int(item)] = float(gate_val)

                    topk = i.detach().cpu().tolist()
                    hits = 0

                    for u_pos, items in enumerate(topk):
                        s = local_sets[u_pos]
                        hits += sum(1 for it in items if it in s)

                    self._usage_local_hits += hits
                    self._usage_total += (bsz * k)

                    if i.size(1) >= 20:
                        for u_pos, items in enumerate(topk):
                            items20 = items[:20]
                            s = local_sets[u_pos]

                            local_hits20 = 0
                            local_high_gate_hits20 = 0

                            for it in items20:
                                if it in s:
                                    local_hits20 += 1

                                    if gate_maps is not None:
                                        gate_val = gate_maps[u_pos].get(int(it), None)

                                        if gate_val is not None:
                                            self._topk_gate_sum_at20 += gate_val
                                            self._topk_gate_count_at20 += 1

                                            if gate_val > 0.5:
                                                local_high_gate_hits20 += 1

                            self._topk_local_hits_at20 += local_hits20
                            self._topk_local_high_gate_hits_at20 += local_high_gate_hits20

        items_ratings_pair = [
            list(zip(
                map(self._data.private_items.get, u_list[0]),
                u_list[1]
            )) for u_list in list(zip(i.detach().cpu().numpy(), v.detach().float().cpu().numpy()))
        ]

        user_ids = [self._data.private_users[u] for u in idx.tolist()]

        return dict(zip(user_ids, items_ratings_pair))
    
    def _extract_topk_item_sets(self, rec_dict, k: int = 20):
        out = {}

        for user, ranked_list in (rec_dict or {}).items():
            topk_items = [item for item, _ in ranked_list[:k]]
            out[user] = set(topk_items)

        return out
    
    def _mean_jaccard_at_k(self, recs_a, recs_b, k: int = 20):
        sets_a = self._extract_topk_item_sets(recs_a, k=k)
        sets_b = self._extract_topk_item_sets(recs_b, k=k)

        common_users = set(sets_a.keys()) & set(sets_b.keys())

        if not common_users:
            return float("nan")

        jaccard_scores = []

        for user in common_users:
            a = sets_a[user]
            b = sets_b[user]
            union = a | b

            if len(union) == 0:
                jaccard_scores.append(1.0)
            else:
                jaccard_scores.append(len(a & b) / len(union))

        return float(np.mean(jaccard_scores))

    def _rng_snapshot(self):
        snap = {
            "py": random.getstate(),
            "np": np.random.get_state(),
            "torch": torch.get_rng_state()
        }

        if torch.cuda.is_available():
            snap["cuda"] = torch.cuda.get_rng_state_all()
        else:
            snap["cuda"] = None
        
        return snap

    def _rng_restore(self, snap):
        if snap is None:
            return

        random.setstate(snap["py"])
        np.random.set_state(snap["np"])

        cpu_rng_state = snap["torch"]

        if not torch.is_tensor(cpu_rng_state):
            cpu_rng_state = torch.tensor(cpu_rng_state, dtype=torch.uint8)

        cpu_rng_state = cpu_rng_state.detach().cpu().to(torch.uint8)
        torch.set_rng_state(cpu_rng_state)

        cuda_rng_states = snap.get("cuda", None)

        if cuda_rng_states is not None:
            normalized_cuda_states = []

            for state in cuda_rng_states:
                if not torch.is_tensor(state):
                    state = torch.tensor(state, dtype=torch.uint8)

                normalized_cuda_states.append(state.detach().cpu().to(torch.uint8))

            torch.cuda.set_rng_state_all(normalized_cuda_states)
    
    def _atomic_torch_save(self, payload, filepath: str):
        tmp_filepath = f"{filepath}.tmp"
        torch.save(payload, tmp_filepath)
        os.replace(tmp_filepath, filepath)

    def _get_early_stopping_runtime_state(self):
        state = {}

        if hasattr(self._early_stopping, "_best"):
            state["_best"] = self._early_stopping._best

        if hasattr(self._early_stopping, "_bad_epochs"):
            state["_bad_epochs"] = self._early_stopping._bad_epochs

        return state

    def _set_early_stopping_runtime_state(self, state):
        for attr in ("_best", "_bad_epochs"):
            if hasattr(self._early_stopping, attr):
                delattr(self._early_stopping, attr)

        if not state:
            return

        if "_best" in state:
            self._early_stopping._best = state["_best"]

        if "_bad_epochs" in state:
            self._early_stopping._bad_epochs = state["_bad_epochs"]

    def _build_resume_checkpoint(self, last_completed_epoch: int):
        return {
            "checkpoint_type": "resume",
            "last_completed_epoch": int(last_completed_epoch),
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._model.optimizer.state_dict(),
            "losses": list(self._losses),
            "results": list(self._results),
            "best_metric_value": self.best_metric_value,
            "best_iteration": getattr(self._params, "best_iteration", None),
            "early_stopping_state": self._get_early_stopping_runtime_state(),
            "rng_state": self._rng_snapshot(),
            "seed": self._seed,
            "epochs": self._epochs,
            "validation_rate": self._validation_rate
        }

    def _save_resume_checkpoint(self, last_completed_epoch: int):
        if not self._save_resume:
            return

        payload = self._build_resume_checkpoint(last_completed_epoch=last_completed_epoch)
        self._atomic_torch_save(payload, self._resume_saving_filepath)

        self.logger.info(
            f"Resume checkpoint saved at '{self._resume_saving_filepath}' "
            f"(last_completed_epoch={last_completed_epoch + 1})."
        )
    
    def _weighted_add(self, store, name, value, weight):
        bucket = store.setdefault(name, {"sum": 0.0, "weight": 0.0})
        bucket["sum"] += float(value) * float(weight)
        bucket["weight"] += float(weight)

    def _weighted_mean(self, store, name):
        bucket = store.get(name, None)

        if bucket is None or bucket["weight"] == 0:
            return float("nan")

        return bucket["sum"] / bucket["weight"]

    def restore_training_state(self):
        if not self._resume_saving_filepath:
            self.logger.warning("Restore requested but no resume filepath is configured. Starting from scratch.")
            return False

        if not os.path.isfile(self._resume_saving_filepath):
            self.logger.info(
                f"No resume checkpoint found at '{self._resume_saving_filepath}'. "
                f"Starting training from scratch."
            )

            return False

        try:
            checkpoint = torch.load(
                self._resume_saving_filepath,
                map_location=self.device,
                weights_only=False
            )

            if checkpoint.get("checkpoint_type") != "resume":
                self.logger.warning(
                    f"File '{self._resume_saving_filepath}' is not a resume checkpoint. "
                    f"Starting from scratch."
                )

                return False

            self._model.load_state_dict(checkpoint["model_state_dict"])
            self._model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            self._losses = list(checkpoint.get("losses", []))
            self._results = list(checkpoint.get("results", []))

            self.best_metric_value = checkpoint.get("best_metric_value", 0)

            best_iteration = checkpoint.get("best_iteration", None)

            if best_iteration is not None:
                self._params.best_iteration = best_iteration

            self._set_early_stopping_runtime_state(checkpoint.get("early_stopping_state", {}))

            rng_state = checkpoint.get("rng_state", None)

            if rng_state is not None:
                self._rng_restore(rng_state)

            last_completed_epoch = int(checkpoint.get("last_completed_epoch", -1))
            self._start_epoch = last_completed_epoch + 1

            self.logger.info(
                f"Resume checkpoint correctly restored from '{self._resume_saving_filepath}'. "
                f"Training will restart from epoch {self._start_epoch + 1}."
            )

            return True

        except Exception as ex:
            self.logger.exception(
                f"Resume checkpoint found but restore failed from "
                f"'{self._resume_saving_filepath}': {ex}"
            )

            raise

    def evaluate(self, it=None, loss=0):
        run_eval = (it is None) or (not (it + 1) % self._validation_rate)

        if not run_eval:
            return

        rng = self._rng_snapshot()

        try:
            k_need = self.evaluator.get_needed_recommendations()

            val_recs = self._collect_recommendations_for_split(
                "validation",
                k_need,
                inference_mode="full",
                count_usage=False,
                fixed_gate_value=None
            )

            result_dict = self.evaluator.eval_validation(val_recs)

            self._losses.append(loss)
            self._results.append(result_dict)

            if it is not None:
                self.logger.info(f"Epoch {(it + 1)}/{self._epochs} eval_loss={loss:.5f}")
            else:
                self.logger.info("Finished validation phase")

            if (len(self._results) - 1) == self.get_best_arg():
                if it is not None:
                    self._params.best_iteration = it + 1

                self.logger.info("******************************************")
                self.best_metric_value = self._results[-1][self._validation_k]["val_results"][self._validation_metric]

                if self._save_weights:
                    if hasattr(self, "_model"):
                        best_payload = {
                            "checkpoint_type": "best",
                            "model_state_dict": self._model.state_dict(),
                            "optimizer_state_dict": self._model.optimizer.state_dict(),
                            "best_metric_value": self.best_metric_value,
                            "best_iteration": getattr(self._params, "best_iteration", None)
                        }

                        self._atomic_torch_save(best_payload, self._saving_filepath)
                    else:
                        self.logger.warning("Saving weights FAILED. No model to save.")
        finally:
            self._rng_restore(rng)
    
    def _run_final_test(self):
        rng = self._rng_snapshot()

        try:
            k_need = self.evaluator.get_needed_recommendations()
            jaccard_k = 20

            results_by_mode = {}
            recs_by_mode = {}

            if not self._inference_for_ranking:
                raise ValueError("inference_for_ranking must contain at least one mode for final test.")

            for mode in self._inference_for_ranking:
                test_recs = self._collect_recommendations_for_split(
                    "test",
                    k_need,
                    inference_mode=mode,
                    count_usage=True,
                    fixed_gate_value=None
                )

                recs_by_mode[mode] = test_recs
                results_by_mode[mode] = self.evaluator.eval_test(test_recs)

                if getattr(self, "_usage_total", 0) > 0:
                    local_share = self._usage_local_hits / self._usage_total
                    global_share = 1.0 - local_share

                    self.logger.info(
                        f"[BRANCH USAGE@{k_need}, mode={mode}] "
                        f"local_topk_share={local_share:.4f}; global_topk_share={global_share:.4f}"
                    )
                
                if mode == "full":
                    self._log_gate_topk_stats(prefix=f"[GATE TOPK@20, mode={mode}]")

            jaccard_full_local = None
            jaccard_full_global = None

            if "full" in recs_by_mode and "local" in recs_by_mode:
                jaccard_full_local = self._mean_jaccard_at_k(
                    recs_by_mode["full"],
                    recs_by_mode["local"],
                    k=jaccard_k
                )

            if "full" in recs_by_mode and "global" in recs_by_mode:
                jaccard_full_global = self._mean_jaccard_at_k(
                    recs_by_mode["full"],
                    recs_by_mode["global"],
                    k=jaccard_k
                )

            jaccard_parts = []

            if jaccard_full_local is not None:
                jaccard_parts.append(f"Jaccard@{jaccard_k}(Full, Local)={jaccard_full_local:.4f}")

            if jaccard_full_global is not None:
                jaccard_parts.append(f"Jaccard@{jaccard_k}(Full, Global)={jaccard_full_global:.4f}")

            if jaccard_parts:
                self.logger.info("[RANKING OVERLAP - TEST] " + "; ".join(jaccard_parts))

            primary = self._primary_inference_mode

            self._final_test_results = results_by_mode[primary]
            self._final_test_results_by_mode = results_by_mode
            self._final_test_recommendations = recs_by_mode[primary]

            if self._save_recs and self._final_test_recommendations is not None:
                self.logger.info(f"Writing final test recommendations at: {self._config.path_output_rec_result}")
                
                store_recommendation(
                    self._final_test_recommendations,
                    os.path.abspath(os.sep.join([self._config.path_output_rec_result, f"{self.name}.tsv"]))
                )
        finally:
            self._rng_restore(rng)

    def restore_best_weights(self, filepath: str = None):
        restore_filepath = filepath if filepath is not None else self._saving_filepath

        if not restore_filepath:
            self.logger.warning("Best-weights restore requested but no saving filepath is configured.")
            return False

        if not os.path.isfile(restore_filepath):
            self.logger.info(f"No best checkpoint found at '{restore_filepath}'.")
            return False

        try:
            checkpoint = torch.load(
                restore_filepath,
                map_location=self.device,
                weights_only=False
            )

            self._model.load_state_dict(checkpoint["model_state_dict"])

            if "optimizer_state_dict" in checkpoint:
                self._model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if "best_metric_value" in checkpoint:
                self.best_metric_value = checkpoint["best_metric_value"]

            best_iteration = checkpoint.get("best_iteration", None)

            if best_iteration is not None:
                self._params.best_iteration = best_iteration

            self.logger.info(f"Best checkpoint correctly restored from '{restore_filepath}'")
            
            return True
        except Exception as ex:
            self.logger.exception(f"Best checkpoint found but restore failed from '{restore_filepath}': {ex}")
            raise

    def restore_weights(self):
        return self.restore_best_weights()