#!/usr/bin/env python3

import argparse
import math
import re
from copy import deepcopy
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Optional, Tuple

VALID_BLOCKS = {
    "baseline",
    "backbone",
    "fusion",
    "contrastive",
    "negatives",
    "structure"
}

BLOCK_LOG_DIRS = {
    "baseline": Path("experiment_logs/00_shared_baseline"),
    "structure": Path("experiment_logs/01_main_structural_analysis"),
    "backbone": Path("experiment_logs/02_backbone_training_recipe"),
    "fusion": Path("experiment_logs/03_fusion_mechanism"),
    "contrastive": Path("experiment_logs/04_contrastive_learning"),
    "negatives": Path("experiment_logs/05_hard_negative_mining")
}

DATASETS = ("amazon-book", "gowalla", "yelp-2018")
BACKBONE_GNNS = ("LightGCN", "NGCF")
CONTRASTIVE_METHODS = ("sgl", "xsimgcl")
NEGATIVE_STRATEGIES = ("uniform", "global_model_aware", "local_model_aware")

RANKING_MODES = ("full", "global", "local")
RANKING_METRICS = ("Recall@20", "nDCG@20")

BRANCH_DIAGNOSTICS = (
    "local_topk_share",
    "global_topk_share",
    "Jaccard@20(Full, Local)",
    "Jaccard@20(Full, Global)"
)

CANDIDATE_DIAGNOSTICS = (
    "local_items_per_user_mean_test",
    "positive_coverage",
    "UB@20",
    "zero_positive_coverage_ratio"
)

BASELINE_BEST_EPOCH_DIAGNOSTICS = (
    "override_ratio",
    "local_items_per_user_mean",
    "delta_override_mean",
    "delta_override_std"
)

FUSION_GATE_DIAGNOSTICS_NO_TEMP = (
    "gate_mean",
    "gate_std",
    "gate_min",
    "gate_max",
    "gate_entropy_mean"
)

FUSION_GATE_TEMPERATURE_DIAGNOSTIC = (
    "gate_temperature"
)

FUSION_TOPK_DIAGNOSTICS = (
    "topk_local_gate_mean@20",
    "topk_local_with_gate_high_share@20"
)

FUSION_LEARNED_TRAINING_DIAGNOSTICS = (
    "override_ratio",
    "local_items_per_user_mean",
    "delta_override_mean",
    "delta_override_std"
)

CONTRASTIVE_COMMON_DIAGNOSTICS = (
    "loss_sup",
    "loss_cl",
    "loss_total",
    "cl_to_sup_ratio",
    "pos_sim_mean",
    "pos_sim_std",
    "neg_sim_mean",
    "neg_sim_std",
    "hardest_neg_sim_mean",
    "alignment_mean",
    "uniformity_mean",
    "user_emb_std_per_dim_mean"
)

SGL_DIAGNOSTICS = (
    "edge_drop_rate_view1",
    "edge_drop_rate_view2",
    "kept_edges_view1",
    "kept_edges_view2",
    "retained_edge_overlap_ratio",
    "isolated_users_ratio_view1",
    "isolated_users_ratio_view2"
)

XSIMGCL_DIAGNOSTICS = (
    "noise_norm_mean",
    "noise_norm_std",
    "noise_to_signal_ratio_mean",
    "clean_noisy_cos_mean",
    "cl_layer",
    "cl_layer_pos_sim_mean",
    "cl_layer_neg_sim_mean"
)

NEGATIVE_COMMON_DIAGNOSTICS = (
    "loss_pos",
    "loss_neg",
    "loss_total",
    "selected_neg_score_mean",
    "selected_neg_score_std",
    "pos_neg_margin_mean",
    "pos_neg_margin_std",
    "hard_ratio_applied",
    "fallback_uniform_ratio",
    "users_without_hard_candidates_ratio",
    "selected_neg_popularity_percentile_mean",
    "duplicate_neg_ratio"
)

GLOBAL_NEGATIVE_DIAGNOSTICS = (
    "selected_neg_emb_score_mean",
    "selected_neg_outside_local_share",
    "global_hard_pool_size_mean"
)

LOCAL_NEGATIVE_DIAGNOSTICS = (
    "local_hard_pool_size_mean",
    "users_with_empty_local_pool_ratio",
    "selected_neg_local_rank_mean"
)

BASELINE_DIAGNOSTICS = BRANCH_DIAGNOSTICS + (
    "positive_coverage",
    "UB@20",
    "zero_positive_coverage_ratio"
) + BASELINE_BEST_EPOCH_DIAGNOSTICS

BACKBONE_DIAGNOSTICS = BRANCH_DIAGNOSTICS

STRUCTURE_DIAGNOSTICS = BRANCH_DIAGNOSTICS + CANDIDATE_DIAGNOSTICS

DISPLAY_NAMES = {
    "local_topk_share": "Local top-K share",
    "global_topk_share": "Global top-K share",
    "Jaccard@20(Full, Local)": "Jaccard@20(Full, Local)",
    "Jaccard@20(Full, Global)": "Jaccard@20(Full, Global)",
    "local_items_per_user_mean_test": "local_items_per_user_mean (TEST)",
    "positive_coverage": "Positive coverage",
    "UB@20": "UB@20",
    "zero_positive_coverage_ratio": "Zero positive coverage ratio",
    "override_ratio": "override_ratio",
    "local_items_per_user_mean": "Local items/user",
    "delta_override_mean": "delta_override_mean",
    "delta_override_std": "delta_override_std",
    "gate_mean": "gate_mean",
    "gate_std": "gate_std",
    "gate_min": "gate_min",
    "gate_max": "gate_max",
    "gate_entropy_mean": "gate_entropy_mean",
    "gate_temperature": "gate_temperature",
    "topk_local_gate_mean@20": "topk_local_gate_mean@20",
    "topk_local_with_gate_high_share@20": "topk_local_with_gate_high_share@20",
    "loss_sup": "loss_sup",
    "loss_cl": "loss_cl",
    "loss_total": "loss_total",
    "cl_to_sup_ratio": "cl_to_sup_ratio",
    "pos_sim_mean": "pos_sim_mean",
    "pos_sim_std": "pos_sim_std",
    "neg_sim_mean": "neg_sim_mean",
    "neg_sim_std": "neg_sim_std",
    "hardest_neg_sim_mean": "hardest_neg_sim_mean",
    "alignment_mean": "alignment_mean",
    "uniformity_mean": "uniformity_mean",
    "user_emb_std_per_dim_mean": "user_emb_std_per_dim_mean",
    "edge_drop_rate_view1": "edge_drop_rate_view1",
    "edge_drop_rate_view2": "edge_drop_rate_view2",
    "kept_edges_view1": "kept_edges_view1",
    "kept_edges_view2": "kept_edges_view2",
    "retained_edge_overlap_ratio": "retained_edge_overlap_ratio",
    "isolated_users_ratio_view1": "isolated_users_ratio_view1",
    "isolated_users_ratio_view2": "isolated_users_ratio_view2",
    "noise_norm_mean": "noise_norm_mean",
    "noise_norm_std": "noise_norm_std",
    "noise_to_signal_ratio_mean": "noise_to_signal_ratio_mean",
    "clean_noisy_cos_mean": "clean_noisy_cos_mean",
    "cl_layer": "cl_layer",
    "cl_layer_pos_sim_mean": "cl_layer_pos_sim_mean",
    "cl_layer_neg_sim_mean": "cl_layer_neg_sim_mean",
    "loss_pos": "loss_pos",
    "loss_neg": "loss_neg",
    "selected_neg_score_mean": "Selected neg. score mean",
    "selected_neg_score_std": "selected_neg_score_std",
    "pos_neg_margin_mean": "pos_neg_margin_mean",
    "pos_neg_margin_std": "pos_neg_margin_std",
    "hard_ratio_applied": "hard_ratio_applied",
    "fallback_uniform_ratio": "fallback_uniform_ratio",
    "users_without_hard_candidates_ratio": "users_without_hard_candidates_ratio",
    "selected_neg_popularity_percentile_mean": "Selected neg. pop. percentile",
    "duplicate_neg_ratio": "duplicate_neg_ratio",
    "selected_neg_emb_score_mean": "selected_neg_emb_score_mean",
    "selected_neg_outside_local_share": "selected_neg_outside_local_share",
    "global_hard_pool_size_mean": "global_hard_pool_size_mean",
    "local_hard_pool_size_mean": "local_hard_pool_size_mean",
    "users_with_empty_local_pool_ratio": "users_with_empty_local_pool_ratio",
    "selected_neg_local_rank_mean": "selected_neg_local_rank_mean"
}

FLOAT_RE = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
NUMBER_RE = rf"(?:{FLOAT_RE}|nan|NaN)"

def clean_value(value: str) -> str:
    value = value.strip().rstrip(",")

    if value.startswith("np.float64(") and value.endswith(")"):
        value = value[len("np.float64("):-1]
    
    return value

def as_number_or_string(value: str) -> Any:
    value = clean_value(value)

    if value in {"True", "False"}:
        return value == "True"
    
    if value.startswith("[") and value.endswith("]"):
        return value
    
    if value.startswith("(") and value.endswith(")"):
        return value
    
    try:
        if re.fullmatch(r"[-+]?\d+", value):
            return int(value)
        
        if re.fullmatch(FLOAT_RE, value):
            return float(value)
    except Exception:
        pass

    return value.strip("'\"")

def extract_float(text: str, key: str) -> Optional[float]:
    match = re.search(rf"{re.escape(key)}=({NUMBER_RE})", text)

    if match is None:
        return None
    
    return float(match.group(1))

def extract_key_values(text: str) -> Dict[str, float]:
    values: Dict[str, float] = {}

    for key, value in re.findall(rf"([A-Za-z0-9_@]+)=({NUMBER_RE})", text):
        try:
            values[key] = float(value)
        except ValueError:
            continue
    
    return values

def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        raise ValueError("Cannot compute mean/std on an empty list.")
    
    if any(value is None for value in values):
        raise ValueError(f"Missing value found: {values}")
    
    if any(isinstance(value, float) and math.isnan(value) for value in values):
        return math.nan, math.nan
    
    if len(values) == 1:
        return values[0], 0.0
    
    return mean(values), stdev(values)

def fmt(values: List[float]) -> str:
    avg, sd = mean_std(values)
    return f"{avg:.6f} ± {sd:.6f}"

def normalize_lr(value: Any) -> float:
    return round(float(value), 12)

def normalize_neigh(value: Any) -> Tuple[int, ...]:
    if value is None:
        return tuple()
    
    return tuple(int(x) for x in re.findall(r"\d+", str(value)))

def config_tuple(config: Dict[str, Any], keys: Iterable[str]) -> Tuple[Any, ...]:
    out = []

    for key in keys:
        value = config.get(key)

        if key == "lr" and value is not None:
            value = normalize_lr(value)
        elif key == "neigh" and value is not None:
            value = normalize_neigh(value)
        elif isinstance(value, float):
            value = round(value, 12)
        
        out.append(value)

    return tuple(out)

def config_to_string(config: Dict[str, Any], keys: Iterable[str]) -> str:
    parts = []

    for key in keys:
        if key not in config:
            continue

        value = config[key]

        if key == "neigh":
            value = "(" + ",".join(str(x) for x in normalize_neigh(value)) + ")"
        
        parts.append(f"{key}={value}")

    return ", ".join(parts)

def list_txt_files(directory: Path, recursive: bool = False) -> List[Path]:
    iterator = directory.rglob("*.txt") if recursive else directory.glob("*.txt")
    return sorted(path for path in iterator if path.is_file())

def infer_dataset_from_path(path: Path, block_dir: Path) -> str:
    try:
        rel = path.relative_to(block_dir)

        if rel.parts and rel.parts[0] in DATASETS:
            return rel.parts[0]
    except Exception:
        pass

    for dataset in DATASETS:
        if dataset in path.name:
            return dataset
        
    return path.stem

def parse_seed_list_from_filename(path: Path) -> Optional[List[int]]:
    match = re.search(r"seed_\[([0-9,\s]+)\]", path.name)

    if match:
        return [int(x.strip()) for x in match.group(1).split(",") if x.strip()]
    
    match = re.search(r"seed_(\d+)", path.name)

    if match:
        return [int(match.group(1))]
    
    return None

def split_runs(lines: List[str]) -> List[List[str]]:
    raw_starts = sorted({
        idx
        for idx, line in enumerate(lines)
        if "Loading parameters" in line
        or re.search(r"Exploration for seed\. Value extracted:\s*\d+", line)
    })

    if not raw_starts:
        raw_starts = [0]

    clusters: List[List[int]] = []

    for idx in raw_starts:
        if not clusters or idx - clusters[-1][-1] > 50:
            clusters.append([idx])
        else:
            clusters[-1].append(idx)

    starts = [min(cluster) for cluster in clusters]

    runs: List[List[str]] = []

    for pos, start_idx in enumerate(starts):
        end_idx = starts[pos + 1] if pos + 1 < len(starts) else len(lines)
        run = lines[start_idx:end_idx]

        if any("Best checkpoint correctly restored" in line for line in run) or any("[EPOCH " in line for line in run):
            runs.append(run)
    
    return runs

def parse_params_from_lines(run_lines: List[str]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    parameter_re = re.compile(r"Parameter\s+([A-Za-z0-9_@]+)\s+set to\s+(.+)$")
    plain_re = re.compile(r"\b([A-Za-z0-9_@]+)\s+set to\s+(.+)$")

    for line in run_lines:
        match = parameter_re.search(line)

        if match:
            params[match.group(1)] = as_number_or_string(match.group(2))
            continue

        match = plain_re.search(line)

        if match:
            key = match.group(1)

            if key not in {"Parameter"}:
                params.setdefault(key, as_number_or_string(match.group(2)))

    return params

def parse_config_from_filename(path: Path) -> Dict[str, Any]:
    name = path.name
    config: Dict[str, Any] = {}

    for gnn in BACKBONE_GNNS:
        if gnn in name:
            config["gnn"] = gnn
            break

    if "xsimgcl" in name.lower():
        config["cl_method"] = "xsimgcl"
    elif "sgl" in name.lower():
        config["cl_method"] = "sgl"

    for strategy in NEGATIVE_STRATEGIES:
        if strategy in name:
            config["negative_sampling_strategy"] = strategy
            break

    match = re.search(r"n_layers_(\d+)", name)

    if match:
        config["n_layers"] = int(match.group(1))

    match = re.search(r"neigh_\(([^)]*)\)", name)

    if match:
        config["neigh"] = normalize_neigh(match.group(1))

    match = re.search(r"lr_([0-9]+(?:\.[0-9]+)?(?:e[-+]?\d+)?)", name, re.IGNORECASE)

    if match:
        config["lr"] = float(match.group(1))

    match = re.search(r"fgv_(v\d+)", name)

    if match:
        config["fusion_gate_version"] = match.group(1)

    match = re.search(r"lambda_cl_([0-9]+(?:\.[0-9]+)?)", name)

    if match:
        config["lambda_cl"] = float(match.group(1))

    match = re.search(r"edge_drop_p_([0-9]+(?:\.[0-9]+)?)", name)

    if match:
        config["edge_drop_p"] = float(match.group(1))

    match = re.search(r"xsim_eps_([0-9]+(?:\.[0-9]+)?)", name)

    if match:
        config["xsim_eps"] = float(match.group(1))

    match = re.search(r"hard_ratio_([0-9]+(?:\.[0-9]+)?)", name)

    if match:
        config["hard_ratio"] = float(match.group(1))

    match = re.search(r"feature_type_([A-Za-z0-9_]+)_", name)

    if match:
        config["feature_type"] = match.group(1)

    if "fixed_gates" in path.parts or name.startswith("fixed_gates"):
        config["fixed_gates"] = True

    if "SANITY_CHECK" in name:
        config["analysis_block"] = "sanity_check"
    elif "feature_type_const_original" in name:
        config["analysis_block"] = "sampling"
    else:
        config.setdefault("analysis_block", "features")

    return config

def parse_seed_from_run(run_lines: List[str]) -> Optional[int]:
    for line in run_lines:
        match = re.search(r"Exploration for seed\. Value extracted:\s*(\d+)", line)

        if match:
            return int(match.group(1))
    
    params = parse_params_from_lines(run_lines)
    
    if "seed" in params:
        return int(float(params["seed"]))
    
    for line in run_lines:
        match = re.search(r"(?:^|\s)seed\s+set to\s+(\d+)", line)
        
        if match:
            return int(match.group(1))
    
    return None

def build_config(run_lines: List[str], path: Path) -> Dict[str, Any]:
    config = parse_config_from_filename(path)
    params = parse_params_from_lines(run_lines)
    config.update(params)

    if "neigh" in config:
        config["neigh"] = normalize_neigh(config["neigh"])
    
    for key in ("lr", "lambda_cl", "tau_cl", "edge_drop_p", "xsim_eps", "hard_ratio", "gate_temperature"):
        if key in config:
            try:
                config[key] = float(config[key])
            except Exception:
                pass
    
    for key in ("n_layers", "seed", "best_iteration"):
        if key in config:
            try:
                config[key] = int(float(config[key]))
            except Exception:
                pass
    
    if "fusion_gate_version" in config:
        config["fusion_gate_version"] = str(config["fusion_gate_version"])
    
    if "cl_method" in config:
        config["cl_method"] = str(config["cl_method"]).lower()
    
    if "negative_sampling_strategy" in config:
        config["negative_sampling_strategy"] = str(config["negative_sampling_strategy"])
    
    return config

def find_restore_index(run_lines: List[str]) -> Optional[int]:
    for idx, line in enumerate(run_lines):
        if "Best checkpoint correctly restored" in line:
            return idx
    
    return None

def parse_validation_history(run_lines: List[str]) -> List[Tuple[int, float]]:
    restore_idx = find_restore_index(run_lines)
    train_lines = run_lines[:restore_idx] if restore_idx is not None else run_lines
    current_epoch: Optional[int] = None
    history: List[Tuple[int, float]] = []

    for line in train_lines:
        epoch_match = re.search(r"Epoch\s+(\d+)/\d+\s+eval_loss", line)
        
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            continue
        if current_epoch is not None:
            recall_match = re.search(rf"\bRecall\s+({FLOAT_RE})", line)
            
            if recall_match:
                history.append((current_epoch, float(recall_match.group(1))))
                current_epoch = None
    
    return history

def parse_best_epoch(run_lines: List[str], config: Dict[str, Any]) -> Tuple[Optional[int], Optional[float]]:
    history = parse_validation_history(run_lines)
    
    if not history:
        return None, None
    
    best_iteration = config.get("best_iteration")
    
    if best_iteration is not None:
        for epoch, recall in history:
            if epoch == best_iteration:
                return epoch, recall
    
    return max(history, key=lambda item: (item[1], -item[0]))

def parse_epoch_diagnostics(run_lines: List[str], best_epoch: Optional[int]) -> Dict[str, float]:
    if best_epoch is None:
        return {}
    
    pattern = re.compile(rf"\[EPOCH\s+{best_epoch}/\d+\]")
    
    for line in run_lines:
        if pattern.search(line):
            return extract_key_values(line)
    
    return {}

def ensure_test(tests: Dict[str, Dict[str, Any]], key: str) -> Dict[str, Any]:
    if key not in tests:
        tests[key] = {
            "ranking": {mode: {} for mode in RANKING_MODES},
            "diagnostics": {}
        }

    return tests[key]

def fixed_key(value: Optional[str]) -> str:
    if value is None:
        return "default"
    
    return f"fixed_g={float(value):.2f}"

def parse_test_sections(run_lines: List[str]) -> Dict[str, Dict[str, Any]]:
    restore_idx = find_restore_index(run_lines)

    if restore_idx is None:
        return {}
    
    tail = run_lines[restore_idx + 1:]

    tests: Dict[str, Dict[str, Any]] = {}
    current_key = "default"
    current_mode: Optional[str] = None
    branch_waiting_for_test = False
    test_open_without_mode = False
    active_test_key: Optional[str] = None
    active_test_mode: Optional[str] = None
    pending_ranking: Dict[str, float] = {}

    branch_re = re.compile(
        rf"\[BRANCH USAGE@20(?:, fixed_g=({FLOAT_RE}))?, mode=(full|global|local)\]\s+"
        rf"local_topk_share=({FLOAT_RE});\s+global_topk_share=({FLOAT_RE})"
    )

    gate_topk_re = re.compile(
        rf"\[GATE TOPK@20(?:, fixed_g=({FLOAT_RE}))?, mode=(full|global|local)\]\s+"
        rf"topk_local_gate_mean@20=({FLOAT_RE});\s+topk_local_with_gate_high_share@20=({FLOAT_RE})"
    )

    overlap_standard_re = re.compile(
        rf"\[RANKING OVERLAP - TEST\]\s+Jaccard@20\(Full, Local\)=({FLOAT_RE});\s+"
        rf"Jaccard@20\(Full, Global\)=({FLOAT_RE})"
    )

    overlap_fixed_re = re.compile(
        rf"\[RANKING OVERLAP, fixed_g=({FLOAT_RE})\]\s+Jaccard@20\(Full, Local\)=({FLOAT_RE});\s+"
        rf"Jaccard@20\(Full, Global\)=({FLOAT_RE})"
    )

    def flush_pending_to(key: str, mode: str) -> None:
        nonlocal pending_ranking

        if not pending_ranking:
            return
        
        target = ensure_test(tests, key)
        target["ranking"].setdefault(mode, {})
        target["ranking"][mode].update(pending_ranking)
        pending_ranking = {}

    for line in tail:
        if "[CANDIDATE DIAGNOSTICS - TEST@20]" in line:
            target = ensure_test(tests, "default")
            local_items = extract_float(line, "local_items_per_user_mean")
            pos_cov = extract_float(line, "pos_coverage")
            ub = extract_float(line, "UB@20")
            zero = extract_float(line, "users_with_zero_pos_coverage_ratio")
            
            if local_items is not None:
                target["diagnostics"]["local_items_per_user_mean_test"] = local_items
            
            if pos_cov is not None:
                target["diagnostics"]["positive_coverage"] = pos_cov
            
            if ub is not None:
                target["diagnostics"]["UB@20"] = ub
            
            if zero is not None:
                target["diagnostics"]["zero_positive_coverage_ratio"] = zero
            
            continue

        branch_match = branch_re.search(line)

        if branch_match:
            current_key = fixed_key(branch_match.group(1))
            current_mode = branch_match.group(2)
            target = ensure_test(tests, current_key)

            if current_mode == "full":
                target["diagnostics"]["local_topk_share"] = float(branch_match.group(3))
                target["diagnostics"]["global_topk_share"] = float(branch_match.group(4))
            
            if pending_ranking:
                flush_pending_to(current_key, current_mode)
                active_test_key = None
                active_test_mode = None
                test_open_without_mode = False
            elif test_open_without_mode:
                active_test_key = current_key
                active_test_mode = current_mode
                test_open_without_mode = False
                branch_waiting_for_test = False
            else:
                branch_waiting_for_test = True

            continue

        gate_match = gate_topk_re.search(line)

        if gate_match:
            key = fixed_key(gate_match.group(1))
            mode = gate_match.group(2)
            target = ensure_test(tests, key)

            if mode == "full":
                target["diagnostics"]["topk_local_gate_mean@20"] = float(gate_match.group(3))
                target["diagnostics"]["topk_local_with_gate_high_share@20"] = float(gate_match.group(4))
            
            continue

        if "Test Evaluation results" in line:
            if branch_waiting_for_test and current_mode is not None:
                active_test_key = current_key
                active_test_mode = current_mode
            else:
                active_test_key = None
                active_test_mode = None
                test_open_without_mode = True

            branch_waiting_for_test = False
            pending_ranking = {}

            continue

        recall_match = re.search(rf"\bRecall\s+({FLOAT_RE})", line)

        if recall_match and (active_test_mode is not None or active_test_key is None):
            value = float(recall_match.group(1))

            if active_test_key is not None and active_test_mode is not None:
                target = ensure_test(tests, active_test_key)
                target["ranking"][active_test_mode]["Recall@20"] = value
            else:
                pending_ranking["Recall@20"] = value

            continue

        ndcg_match = re.search(rf"\bnDCG\s+({FLOAT_RE})", line)

        if ndcg_match and (active_test_mode is not None or active_test_key is None):
            value = float(ndcg_match.group(1))

            if active_test_key is not None and active_test_mode is not None:
                target = ensure_test(tests, active_test_key)
                target["ranking"][active_test_mode]["nDCG@20"] = value
                active_test_key = None
                active_test_mode = None
                test_open_without_mode = False
            else:
                pending_ranking["nDCG@20"] = value

            continue

        overlap_match = overlap_standard_re.search(line)

        if overlap_match:
            target = ensure_test(tests, "default")
            target["diagnostics"]["Jaccard@20(Full, Local)"] = float(overlap_match.group(1))
            target["diagnostics"]["Jaccard@20(Full, Global)"] = float(overlap_match.group(2))
            
            continue

        overlap_fixed_match = overlap_fixed_re.search(line)

        if overlap_fixed_match:
            key = fixed_key(overlap_fixed_match.group(1))
            target = ensure_test(tests, key)
            target["diagnostics"]["Jaccard@20(Full, Local)"] = float(overlap_fixed_match.group(2))
            target["diagnostics"]["Jaccard@20(Full, Global)"] = float(overlap_fixed_match.group(3))
            
            continue

    return tests

def parse_log_file(
    path: Path,
    default_seed: Optional[int] = None,
    default_seeds: Optional[List[int]] = None
) -> List[Dict[str, Any]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    raw_runs = split_runs(lines)
    parsed: List[Dict[str, Any]] = []

    for idx, run_lines in enumerate(raw_runs):
        config = build_config(run_lines, path)
        seed = parse_seed_from_run(run_lines)

        if seed is None and "seed" in config:
            seed = int(config["seed"])
        
        if seed is None and default_seeds:
            if len(default_seeds) == 1:
                seed = default_seeds[0]
            elif idx < len(default_seeds):
                seed = default_seeds[idx]
        
        if seed is None and default_seed is not None:
            seed = default_seed
        
        if seed is not None:
            config["seed"] = int(seed)

        best_epoch, best_validation_recall = parse_best_epoch(run_lines, config)
        epoch_diagnostics = parse_epoch_diagnostics(run_lines, best_epoch)
        tests = parse_test_sections(run_lines)

        default_test = tests.get("default", {"ranking": {m: {} for m in RANKING_MODES}, "diagnostics": {}})
        diagnostics = dict(default_test.get("diagnostics", {}))
        diagnostics.update(epoch_diagnostics)

        parsed.append({
            "source_file": path,
            "seed": int(seed) if seed is not None else None,
            "best_epoch": best_epoch,
            "best_validation_recall": best_validation_recall,
            "config": config,
            "ranking": default_test.get("ranking", {m: {} for m in RANKING_MODES}),
            "diagnostics": diagnostics,
            "tests": tests
        })

    return parsed

def run_has_ranking(run: Dict[str, Any]) -> bool:
    ranking = run.get("ranking", {})
    return all(metric in ranking.get(mode, {}) for mode in RANKING_MODES for metric in RANKING_METRICS)

def make_test_run(run: Dict[str, Any], test_key: str) -> Dict[str, Any]:
    out = deepcopy(run)
    test = run["tests"][test_key]
    out["ranking"] = test["ranking"]
    diagnostics = dict(test.get("diagnostics", {}))

    for key, value in run.get("diagnostics", {}).items():
        diagnostics.setdefault(key, value)
    
    out["diagnostics"] = diagnostics
    out["config"] = dict(run.get("config", {}))
    
    if test_key.startswith("fixed_g="):
        out["config"]["fixed_g"] = float(test_key.split("=", 1)[1])
    
    return out

def available_diagnostic_keys(runs: List[Dict[str, Any]], requested: Tuple[str, ...]) -> Tuple[List[str], List[str]]:
    available: List[str] = []
    missing: List[str] = []

    for key in requested:
        if all(key in run.get("diagnostics", {}) for run in runs):
            available.append(key)
        else:
            missing.append(key)
    
    return available, missing

def print_summary(
    title: str,
    runs: List[Dict[str, Any]],
    diagnostic_keys: Tuple[str, ...],
    expected_seed_count: Optional[int] = 5
) -> None:
    runs = [run for run in runs if run_has_ranking(run)]
    runs = sorted(runs, key=lambda r: (-1 if r.get("seed") is None else int(r["seed"])))

    print("=" * 130)
    print(title)
    
    if not runs:
        print("WARNING: no complete test-ranking runs found for this group. Skipping aggregation.\n")
        return
    
    print(f"Parsed seeds: {[run.get('seed') for run in runs]}")
    print(f"Best epochs: {[run.get('best_epoch') for run in runs]}")
    
    if expected_seed_count is not None and len(runs) != expected_seed_count:
        print(f"WARNING: expected {expected_seed_count} seeds for a final mean±std aggregation, found {len(runs)}.")

    print("\nRanking metrics on test set")
    print("-" * 130)
    print(f"{'Metric':<12} {'Full':>24} {'Global':>24} {'Local':>24}")
    
    for metric in RANKING_METRICS:
        row = []

        for mode in RANKING_MODES:
            values = [run["ranking"][mode][metric] for run in runs]
            row.append(fmt(values))
        
        print(f"{metric:<12} {row[0]:>24} {row[1]:>24} {row[2]:>24}")

    _, missing = available_diagnostic_keys(runs, diagnostic_keys)

    if missing:
        details = []

        for key in missing:
            missing_seeds = [run.get("seed") for run in runs if key not in run.get("diagnostics", {})]
            details.append(f"{key} missing for seeds {missing_seeds}")
        
        raise ValueError(f"Missing requested diagnostics for group '{title}': " + "; ".join(details))

    print("\nDiagnostics")
    print("-" * 130)
    print(f"{'Diagnostic':<48} {'mean ± std':>24}")

    for key in diagnostic_keys:
        values = [run["diagnostics"][key] for run in runs]
        label = DISPLAY_NAMES.get(key, key)
        print(f"{label:<48} {fmt(values):>24}")

    print()

def expected_seed_warning(runs: List[Dict[str, Any]], context: str) -> None:
    seeds = {run.get("seed") for run in runs}

    if seeds != {0, 1, 2, 3, 4}:
        print(f"WARNING: {context}: expected seeds [0,1,2,3,4], found {sorted(seeds)}")

def group_runs(runs: Iterable[Dict[str, Any]], key_func) -> Dict[Tuple[Any, ...], List[Dict[str, Any]]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    
    for run in runs:
        key = key_func(run)
        groups.setdefault(key, []).append(run)
    
    return groups

################
### baseline ###
################

def run_baseline() -> None:
    block_dir = BLOCK_LOG_DIRS["baseline"]
    
    if not block_dir.exists():
        raise SystemExit(f"Directory not found: {block_dir}")
    
    for path in list_txt_files(block_dir):
        runs = parse_log_file(path, default_seeds=parse_seed_list_from_filename(path))
        print_summary(f"Baseline | Dataset/file: {path.stem}", runs, BASELINE_DIAGNOSTICS)

################
### backbone ###
################

def backbone_config_key(config: Dict[str, Any]) -> Tuple[Any, ...]:
    return config_tuple(config, ("gnn", "n_layers", "neigh", "lr"))

def collect_backbone_screening(dataset_dir: Path) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    
    for path in list_txt_files(dataset_dir):
        for run in parse_log_file(path, default_seed=0):
            config = run["config"]

            if config.get("gnn") in BACKBONE_GNNS and run.get("best_validation_recall") is not None:
                run["seed"] = 0
                run["config"]["seed"] = 0
                runs.append(run)
    
    return runs

def collect_matching_multiseed(dataset_dir: Path, selected_config: Dict[str, Any], key_func) -> List[Dict[str, Any]]:
    multiseed_dir = dataset_dir / "MULTI-SEED"
    
    if not multiseed_dir.exists():
        return []
    
    target_key = key_func(selected_config)
    out: List[Dict[str, Any]] = []
    
    for path in list_txt_files(multiseed_dir):
        for run in parse_log_file(path, default_seeds=parse_seed_list_from_filename(path)):
            if key_func(run["config"]) == target_key:
                out.append(run)
    
    return out

def collect_multiseed_groups(dataset_dir: Path, key_func, predicate=lambda config: True) -> Dict[Tuple[Any, ...], List[Dict[str, Any]]]:
    multiseed_dir = dataset_dir / "MULTI-SEED"
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    
    if not multiseed_dir.exists():
        return groups
    
    for path in list_txt_files(multiseed_dir):
        for run in parse_log_file(path, default_seeds=parse_seed_list_from_filename(path)):
            if not predicate(run["config"]):
                continue

            key = key_func(run["config"])
            groups.setdefault(key, []).append(run)
    
    return groups

def find_seed0_for_selected_config(
    screening_runs: List[Dict[str, Any]],
    selected_key: Tuple[Any, ...],
    key_func,
    fallback_runs: Optional[List[Dict[str, Any]]] = None,
    context: str = ""
) -> Optional[Dict[str, Any]]:
    exact = [run for run in screening_runs if key_func(run["config"]) == selected_key]
    
    if exact:
        return max(exact, key=lambda r: (-1.0 if r.get("best_validation_recall") is None else r["best_validation_recall"]))
    
    if fallback_runs:
        print(f"WARNING: no exact seed-0 screening match for selected MULTI-SEED config in {context}; using best available seed-0 fallback.")
        return max(fallback_runs, key=lambda r: (-1.0 if r.get("best_validation_recall") is None else r["best_validation_recall"]))
    
    print(f"WARNING: no seed-0 run found for selected MULTI-SEED config in {context}.")
    
    return None

def run_backbone() -> None:
    block_dir = BLOCK_LOG_DIRS["backbone"]
    
    if not block_dir.exists():
        raise SystemExit(f"Directory not found: {block_dir}")
    
    for dataset_dir in sorted(path for path in block_dir.iterdir() if path.is_dir()):
        dataset = dataset_dir.name
        screening = collect_backbone_screening(dataset_dir)

        multiseed_groups = collect_multiseed_groups(
            dataset_dir,
            backbone_config_key,
            predicate=lambda cfg: cfg.get("gnn") in BACKBONE_GNNS
        )

        for gnn in BACKBONE_GNNS:
            candidates = [r for r in screening if r["config"].get("gnn") == gnn]
            gnn_groups = {k: v for k, v in multiseed_groups.items() if v and v[0]["config"].get("gnn") == gnn}

            if not gnn_groups:
                if not candidates:
                    print(f"WARNING: no seed-0 or MULTI-SEED runs found for dataset={dataset}, gnn={gnn}")
                    continue
                
                best_seed0 = max(candidates, key=lambda r: r["best_validation_recall"])
                selected = best_seed0["config"]
                gnn_groups = {backbone_config_key(selected): collect_matching_multiseed(dataset_dir, selected, backbone_config_key)}
            
            for selected_key, multiseed in sorted(gnn_groups.items(), key=lambda item: str(item[0])):
                selected = multiseed[0]["config"] if multiseed else candidates[0]["config"]
                
                best_seed0 = find_seed0_for_selected_config(
                    screening, selected_key, backbone_config_key,
                    fallback_runs=candidates,
                    context=f"backbone dataset={dataset}, gnn={gnn}"
                )

                by_seed = {}

                if best_seed0 is not None:
                    by_seed[0] = best_seed0
                
                for run in multiseed:
                    if run.get("seed") is not None:
                        by_seed[int(run["seed"])] = run
                
                final_runs = [by_seed[s] for s in sorted(by_seed)]
                expected_seed_warning(final_runs, f"backbone dataset={dataset}, gnn={gnn}")
                
                title = (
                    f"Backbone | Dataset: {dataset} | GNN: {gnn}\n"
                    f"Selected config from MULTI-SEED/best logs: {config_to_string(selected, ('gnn','n_layers','neigh','lr'))}\n"
                    f"Seed-0 validation Recall@20: {best_seed0['best_validation_recall']:.6f}"
                    if best_seed0 is not None else
                    f"Backbone | Dataset: {dataset} | GNN: {gnn}\n"
                    f"Selected config from MULTI-SEED/best logs: {config_to_string(selected, ('gnn','n_layers','neigh','lr'))}\n"
                    f"Seed-0 validation Recall@20: MISSING"
                )

                print_summary(title, final_runs, BACKBONE_DIAGNOSTICS)

##############
### fusion ###
##############

def fusion_learned_diagnostics(version: str) -> Tuple[str, ...]:
    diagnostics = (
        FUSION_GATE_DIAGNOSTICS_NO_TEMP
        + (FUSION_GATE_TEMPERATURE_DIAGNOSTIC if version == "v3" else tuple())
        + FUSION_TOPK_DIAGNOSTICS
        + FUSION_LEARNED_TRAINING_DIAGNOSTICS
        + BRANCH_DIAGNOSTICS
    )
    
    return tuple(dict.fromkeys(diagnostics))

def fusion_fixed_diagnostics() -> Tuple[str, ...]:
    return tuple(dict.fromkeys(FUSION_TOPK_DIAGNOSTICS + BRANCH_DIAGNOSTICS))

def run_fusion() -> None:
    block_dir = BLOCK_LOG_DIRS["fusion"]
    
    if not block_dir.exists():
        raise SystemExit(f"Directory not found: {block_dir}")

    learned_runs: List[Dict[str, Any]] = []
    fixed_runs: List[Dict[str, Any]] = []

    for dataset_dir in sorted(path for path in block_dir.iterdir() if path.is_dir()):
        dataset = dataset_dir.name
        
        for path in list_txt_files(dataset_dir):
            if "fixed_gates" in path.parts:
                continue
            
            for run in parse_log_file(path, default_seeds=parse_seed_list_from_filename(path)):
                run["dataset"] = dataset

                if run["config"].get("fusion_gate_version"):
                    learned_runs.append(run)

        fixed_dir = dataset_dir / "fixed_gates"

        if fixed_dir.exists():
            for path in list_txt_files(fixed_dir):
                for run in parse_log_file(path, default_seeds=parse_seed_list_from_filename(path)):
                    for test_key in sorted(k for k in run["tests"] if k.startswith("fixed_g=")):
                        fixed_run = make_test_run(run, test_key)
                        fixed_run["dataset"] = dataset
                        fixed_runs.append(fixed_run)

    learned_groups = group_runs(
        learned_runs,
        lambda r: (r["dataset"], r["config"].get("fusion_gate_version"))
    )

    for (dataset, version), runs in sorted(learned_groups.items()):
        diagnostics = fusion_learned_diagnostics(str(version))
        print_summary(f"Fusion | Learned gate | Dataset: {dataset} | version={version}", runs, diagnostics)

    fixed_groups = group_runs(
        fixed_runs,
        lambda r: (r["dataset"], r["config"].get("fixed_g"))
    )

    for (dataset, fixed_g), runs in sorted(fixed_groups.items()):
        diagnostics = fusion_fixed_diagnostics()
        print_summary(f"Fusion | Fixed gate inference | Dataset: {dataset} | fixed_g={fixed_g:.2f}", runs, diagnostics)

###################
### contrastive ###
###################

def contrastive_key(config: Dict[str, Any]) -> Tuple[Any, ...]:
    method = config.get("cl_method")

    if method == "sgl":
        return config_tuple(config, ("cl_method", "lambda_cl", "edge_drop_p"))
    
    if method == "xsimgcl":
        return config_tuple(config, ("cl_method", "lambda_cl", "xsim_eps"))
    
    return config_tuple(config, ("cl_method", "lambda_cl", "edge_drop_p", "xsim_eps"))

def contrastive_diagnostics(method: str) -> Tuple[str, ...]:
    base = CONTRASTIVE_COMMON_DIAGNOSTICS + BRANCH_DIAGNOSTICS
    
    if method == "sgl":
        return base + SGL_DIAGNOSTICS
    
    if method == "xsimgcl":
        return base + XSIMGCL_DIAGNOSTICS
    
    return base

def collect_contrastive_screening(dataset_dir: Path) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    
    for path in list_txt_files(dataset_dir):
        for run in parse_log_file(path, default_seed=0):
            method = run["config"].get("cl_method")

            if method in CONTRASTIVE_METHODS and run.get("best_validation_recall") is not None:
                run["seed"] = 0
                run["config"]["seed"] = 0
                runs.append(run)
    
    return runs

def run_contrastive() -> None:
    block_dir = BLOCK_LOG_DIRS["contrastive"]
    
    if not block_dir.exists():
        raise SystemExit(f"Directory not found: {block_dir}")
    
    for dataset_dir in sorted(path for path in block_dir.iterdir() if path.is_dir()):
        dataset = dataset_dir.name
        screening = collect_contrastive_screening(dataset_dir)

        multiseed_groups = collect_multiseed_groups(
            dataset_dir,
            contrastive_key,
            predicate=lambda cfg: cfg.get("cl_method") in CONTRASTIVE_METHODS
        )

        for method in CONTRASTIVE_METHODS:
            candidates = [r for r in screening if r["config"].get("cl_method") == method]
            method_groups = {k: v for k, v in multiseed_groups.items() if v and v[0]["config"].get("cl_method") == method}
            
            if not method_groups:
                if not candidates:
                    print(f"WARNING: no seed-0 or MULTI-SEED runs found for dataset={dataset}, method={method}")
                    continue
                
                best_seed0 = max(candidates, key=lambda r: r["best_validation_recall"])
                selected = best_seed0["config"]
                method_groups = {contrastive_key(selected): collect_matching_multiseed(dataset_dir, selected, contrastive_key)}
            
            for selected_key, multiseed in sorted(method_groups.items(), key=lambda item: str(item[0])):
                selected = multiseed[0]["config"] if multiseed else candidates[0]["config"]
                
                best_seed0 = find_seed0_for_selected_config(
                    screening, selected_key, contrastive_key,
                    fallback_runs=candidates,
                    context=f"contrastive dataset={dataset}, method={method}"
                )
                
                by_seed = {}
                
                if best_seed0 is not None:
                    by_seed[0] = best_seed0
                
                for run in multiseed:
                    if run.get("seed") is not None:
                        by_seed[int(run["seed"])] = run
                
                final_runs = [by_seed[s] for s in sorted(by_seed)]
                expected_seed_warning(final_runs, f"contrastive dataset={dataset}, method={method}")
                
                title = (
                    f"Contrastive | Dataset: {dataset} | method={method}\n"
                    f"Selected config from MULTI-SEED/best logs: {config_to_string(selected, ('cl_method','lambda_cl','tau_cl','edge_drop_p','xsim_eps'))}\n"
                    f"Seed-0 validation Recall@20: {best_seed0['best_validation_recall']:.6f}"
                    if best_seed0 is not None else
                    f"Contrastive | Dataset: {dataset} | method={method}\n"
                    f"Selected config from MULTI-SEED/best logs: {config_to_string(selected, ('cl_method','lambda_cl','tau_cl','edge_drop_p','xsim_eps'))}\n"
                    f"Seed-0 validation Recall@20: MISSING"
                )

                print_summary(title, final_runs, contrastive_diagnostics(method))

#################
### negatives ###
#################

def negative_key(config: Dict[str, Any]) -> Tuple[Any, ...]:
    return config_tuple(config, ("negative_sampling_strategy", "hard_ratio"))

def negative_diagnostics(strategy: str) -> Tuple[str, ...]:
    base = NEGATIVE_COMMON_DIAGNOSTICS
    
    if strategy == "global_model_aware":
        return base + GLOBAL_NEGATIVE_DIAGNOSTICS + BRANCH_DIAGNOSTICS
    
    if strategy == "local_model_aware":
        return base + LOCAL_NEGATIVE_DIAGNOSTICS + BRANCH_DIAGNOSTICS
    
    return base + BRANCH_DIAGNOSTICS

def collect_negative_screening(dataset_dir: Path) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    
    for path in list_txt_files(dataset_dir):
        if path.name.startswith("uniform"):
            continue
        
        for run in parse_log_file(path, default_seed=0):
            strategy = run["config"].get("negative_sampling_strategy")
            
            if strategy in {"global_model_aware", "local_model_aware"} and run.get("best_validation_recall") is not None:
                run["seed"] = 0
                run["config"]["seed"] = 0
                runs.append(run)
    
    return runs

def run_negatives() -> None:
    block_dir = BLOCK_LOG_DIRS["negatives"]
    
    if not block_dir.exists():
        raise SystemExit(f"Directory not found: {block_dir}")
    
    for dataset_dir in sorted(path for path in block_dir.iterdir() if path.is_dir()):
        dataset = dataset_dir.name

        uniform_files = list((dataset_dir / "MULTI-SEED").glob("uniform*.txt")) if (dataset_dir / "MULTI-SEED").exists() else []
        uniform_files += list(dataset_dir.glob("uniform*.txt"))
        uniform_runs: List[Dict[str, Any]] = []
        
        for path in sorted(uniform_files):
            uniform_runs.extend(parse_log_file(path, default_seeds=parse_seed_list_from_filename(path)))
        
        if uniform_runs:
            print_summary(f"Negatives | Dataset: {dataset} | strategy=uniform", uniform_runs, negative_diagnostics("uniform"))

        screening = collect_negative_screening(dataset_dir)

        multiseed_groups = collect_multiseed_groups(
            dataset_dir,
            negative_key,
            predicate=lambda cfg: cfg.get("negative_sampling_strategy") in {"global_model_aware", "local_model_aware"}
        )

        for strategy in ("global_model_aware", "local_model_aware"):
            candidates = [r for r in screening if r["config"].get("negative_sampling_strategy") == strategy]
            strategy_groups = {k: v for k, v in multiseed_groups.items() if v and v[0]["config"].get("negative_sampling_strategy") == strategy}
            
            if not strategy_groups:
                if not candidates:
                    print(f"WARNING: no seed-0 or MULTI-SEED runs found for dataset={dataset}, strategy={strategy}")
                    continue
                
                best_seed0 = max(candidates, key=lambda r: r["best_validation_recall"])
                selected = best_seed0["config"]
                strategy_groups = {negative_key(selected): collect_matching_multiseed(dataset_dir, selected, negative_key)}
            
            for selected_key, multiseed in sorted(strategy_groups.items(), key=lambda item: str(item[0])):
                selected = multiseed[0]["config"] if multiseed else candidates[0]["config"]
                
                best_seed0 = find_seed0_for_selected_config(
                    screening, selected_key, negative_key,
                    fallback_runs=candidates,
                    context=f"negatives dataset={dataset}, strategy={strategy}"
                )

                by_seed = {}
                
                if best_seed0 is not None:
                    by_seed[0] = best_seed0
                
                for run in multiseed:
                    if run.get("seed") is not None:
                        by_seed[int(run["seed"])] = run
                
                final_runs = [by_seed[s] for s in sorted(by_seed)]
                expected_seed_warning(final_runs, f"negatives dataset={dataset}, strategy={strategy}")
                
                title = (
                    f"Negatives | Dataset: {dataset} | strategy={strategy}\n"
                    f"Selected config from MULTI-SEED/best logs: {config_to_string(selected, ('negative_sampling_strategy','hard_ratio'))}\n"
                    f"Seed-0 validation Recall@20: {best_seed0['best_validation_recall']:.6f}"
                    if best_seed0 is not None else
                    f"Negatives | Dataset: {dataset} | strategy={strategy}\n"
                    f"Selected config from MULTI-SEED/best logs: {config_to_string(selected, ('negative_sampling_strategy','hard_ratio'))}\n"
                    f"Seed-0 validation Recall@20: MISSING"
                )
                
                print_summary(title, final_runs, negative_diagnostics(strategy))

#################
### structure ###
#################

def structure_group_key(run: Dict[str, Any], block_dir: Path) -> Tuple[Any, ...]:
    path = run["source_file"]
    dataset = infer_dataset_from_path(path, block_dir)
    config = run["config"]
    
    return (
        dataset,
        config.get("analysis_block"),
        config.get("feature_type"),
        int(config.get("n_layers")) if config.get("n_layers") is not None else None,
        normalize_neigh(config.get("neigh"))
    )

def run_structure() -> None:
    block_dir = BLOCK_LOG_DIRS["structure"]

    if not block_dir.exists():
        raise SystemExit(f"Directory not found: {block_dir}")
    
    runs: List[Dict[str, Any]] = []
    
    for path in list_txt_files(block_dir, recursive=True):
        runs.extend(parse_log_file(path, default_seeds=parse_seed_list_from_filename(path)))
    
    groups = group_runs(runs, lambda r: structure_group_key(r, block_dir))
    
    for (dataset, analysis_block, feature_type, n_layers, neigh), group in sorted(groups.items(), key=lambda kv: str(kv[0])):
        title = (
            f"Structure | Dataset: {dataset} | block={analysis_block} | "
            f"feature_type={feature_type} | n_layers={n_layers} | neigh={neigh}"
        )

        expected_count = 3 if analysis_block == "sanity_check" else 5
        print_summary(title, group, STRUCTURE_DIAGNOSTICS, expected_seed_count=expected_count)

############
### main ###
############

def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ContextGNN experiment logs.")
    parser.add_argument("experimental_block", choices=sorted(VALID_BLOCKS))
    args = parser.parse_args()

    if args.experimental_block == "baseline":
        run_baseline()
    elif args.experimental_block == "backbone":
        run_backbone()
    elif args.experimental_block == "fusion":
        run_fusion()
    elif args.experimental_block == "contrastive":
        run_contrastive()
    elif args.experimental_block == "negatives":
        run_negatives()
    elif args.experimental_block == "structure":
        run_structure()
    else:
        raise SystemExit(f"Unsupported experimental_block={args.experimental_block}")

if __name__ == "__main__":
    main()