# Beyond Two-Tower, Back to Local: Revisiting ContextGNN for Top-K Recommendation

This repository accompanies the paper **Beyond Two-Tower, Back to Local: Revisiting ContextGNN for Top-K Recommendation**, available at: \<TODO: link>.

Its purpose is to support the reproducibility of the experimental results reported in the paper by documenting the code origin and software requirements, describing the data format and preprocessing workflow, explaining how to configure and launch experiments, linking each experimental grid to its aggregated result file, and exposing the logs, diagnostics, and auxiliary execution notes that are not fully included in the paper.

## Code Origin

The code in this repository is based on the original ContextGNN implementation released by Kumo: [`kumo-ai/ContextGNN`](https://github.com/kumo-ai/ContextGNN).

The implementation follows the PyTorch and PyTorch Geometric code path of the original ContextGNN repository and adapts it to the top-K recommendation setting using the Elliot framework.

## Requirements

The Python dependencies required to run the code can be installed from the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

The provided requirements are configured for the PyTorch/PyG CUDA 12.1 wheels. Systems with a different CUDA setup may require adapting the PyTorch and PyTorch Geometric installation commands accordingly.

## Reproducing the Experiments

The experimental analysis in the paper is organized into different blocks. Each block is implemented in the corresponding repository branch, but the execution workflow is the same across branches: after selecting the branch and the [desired YAML configuration file](#3-configure-the-experiment), experiments are launched through the same training script.

### 1. Recommendation data format

For each dataset, the expected data directory is:

```bash
data/<dataset_name>/
```

For example, for Amazon-Book:

```bash
data/amazon-book/
```

The preprocessing script expects the following raw files:

- `user_list.txt`: user mapping file with columns `org_id` and `remap_id`;
- `item_list.txt`: item mapping file with columns `org_id` and `remap_id`;
- `train.txt`: training interactions, one user per row, followed by the interacted item IDs;
- `test.txt`: test interactions, with the same format as `train.txt`.

All interaction files are expected to use the remapped numerical IDs. The repository already includes the three datasets used in the paper, **Amazon-Book**, **Yelp2018**, and **Gowalla**, in the format required to run the provided experiments.

### 2. Prepare the data for Elliot and RelBench

For **Amazon-Book**, **Yelp2018**, and **Gowalla**, this data-preparation step has already been performed in the repository. Run it only to regenerate the processed files from the raw recommendation data or to prepare a different dataset.

The raw files can be converted into the format required by Elliot and RelBench with:

```bash
python map_relbench.py --dataset <dataset_name>
```

For example, for Amazon-Book:

```bash
python map_relbench.py --dataset amazon-book
```

This step creates the Elliot interaction files used for evaluation, including `train_elliot.tsv`, `val_elliot.tsv`, and `test_elliot.tsv`, together with the RelBench-style tables required by ContextGNN.

The script also builds the validation split from the training data using a fixed seed, applies cold-user and cold-item filtering, and assigns synthetic pseudo-timestamps only to satisfy the RelBench input format. No temporal information is introduced in the classical top-K recommendation setting.

### 3. Configure the experiment

Experiments are controlled by YAML files stored in:

```bash
config_files/
```

The launcher expects ContextGNN configuration files named as:

```bash
contextgnn_<dataset_name>.yml
```

For example, for Amazon-Book:

```bash
config_files/contextgnn_amazon-book.yml
```

### 4. Run the experiment

Once the desired branch and YAML file have been selected, an experiment can be launched with:

```bash
python start_experiments.py --dataset <dataset_name> --model contextgnn
```

For example, for Amazon-Book:

```bash
python start_experiments.py --dataset amazon-book --model contextgnn
```

This command runs the Elliot experiment defined in:

```bash
config_files/contextgnn_amazon-book.yml
```

## Experimental Branches, Grids, and Results

The experiments reported in the paper are organized into separate repository branches. To switch to a specific branch, run:

```bash
git checkout <branch_name>
```

Experimental blocks and branches, ordered as in the paper:

| Branch                 | Experimental block                                                                                                |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `main`                 | Reproduced baseline                                                                                               |
| `structural-analysis`  | Candidate-set bottleneck under neighbor sampling; Gowalla unified constant-encoding sanity check; feature ablations with constant, random, and informative inputs |
| `main`                 | GNN-backbone analysis                                                                                             |
| `fusion-gate`          | Learned and fixed local-global fusion gates                                                                       |
| `contrastive-learning` | SGL-style and XSimGCL-style contrastive learning                                                                  |
| `hard-negatives`       | BCE training and hard negative mining                                                                             |

The original baseline configurations are stored in `config_files/` on the `main` branch. The grids below report only the hyperparameters that must be changed with respect to those baseline YAML files in order to reproduce each experimental block. All other settings, including data split, evaluation protocol, gradient accumulation, early stopping, and Full, Global, and Local evaluation setup, follow the baseline configuration.

### Baseline results

**Branch:** `main`

The baseline runs use the original dataset-specific YAML files in `config_files/` without requiring an additional hyperparameter grid in this README.

**Multi-seed protocol:** direct multi-seed evaluation with `seed ∈ {0,1,2,3,4}`.

Aggregated value tables and diagnostic definitions: `experiment_results/baseline.md`.

### Candidate-set / neighbor-sampling grid

**Branch:** `structural-analysis`

| Hyperparameter                   | Values                                                        |
| -------------------------------- | ------------------------------------------------------------- |
| `feature_type`                   | `const_original`                                              |
| `n_layers`                       | `2`, `4`                                                      |
| `neigh`, 2 layers                | `(4,4)`, `(8,8)`, `(16,16)`                                   |
| `neigh`, 4 layers                | `(4,4,4,4)`, `(8,8,8,8)`, plus the baseline-width setting     |
| Baseline-width `neigh`, 4 layers | Amazon-Book: `(16,16,8,8)`; Yelp2018/Gowalla: `(16,16,16,16)` |

**Multi-seed protocol:** direct multi-seed evaluation with `seed ∈ {0,1,2,3,4}`.

Aggregated value tables and diagnostic definitions: `experiment_results/structure.md`.

### Feature-ablation grid

**Branch:** `structural-analysis`

| Hyperparameter | Values                                   |
| -------------- | ---------------------------------------- |
| `feature_type` | `const_unified`, `random`, `informative` |

For `feature_type=informative`, the additional fixed `Node2Vec` hyperparameters are `embedding_dim=16`, `walk_length=20`, `context_size=10`, `walks_per_node=10`, `num_negative_samples=1`, `p=1.0`, and `q=1.0`. These values correspond to the branch defaults and therefore do not need to be explicitly specified in the YAML configuration.

**Multi-seed protocol:** direct multi-seed evaluation with `seed ∈ {0,1,2,3,4}` for the main feature-ablation rows (`const_unified`, `random`, and `informative`) reported in the paper table.

**Gowalla unified constant-encoding sanity check:** `experiment_results/structure.md` also reports a separate Gowalla sanity-check table comparing the original baseline constant-feature path and the unified dense constant-feature path under narrow and wide neighbor-sampling regimes. The additional narrow unified-constant run uses `feature_type=const_unified`, `n_layers=2`, `neigh=(16,16)`, and seeds `{0,1,2}`; the other rows in the sanity-check table reuse the corresponding five-seed neighbor-sampling, baseline-width, and feature-ablation logs.

Aggregated value tables and diagnostic definitions: `experiment_results/structure.md`.

### GNN-backbone grid

**Branch:** `main`

| Hyperparameter    | Values                                                        |
| ----------------- | ------------------------------------------------------------- |
| `gnn`             | `LightGCN`, `NGCF`                                            |
| `lr`              | `0.0001`, `0.0005`, `0.001`, `0.005`, `0.01`                  |
| `n_layers`        | `1`, `2`, `3`, `4`                                            |
| `neigh`, 1 layer  | `(16,)`                                                       |
| `neigh`, 2 layers | `(16,16)`                                                     |
| `neigh`, 3 layers | Amazon-Book: `(16,16,8)`; Yelp2018/Gowalla: `(16,16,16)`      |
| `neigh`, 4 layers | Amazon-Book: `(16,16,8,8)`; Yelp2018/Gowalla: `(16,16,16,16)` |

**Multi-seed protocol:** seed-0  screening for `LightGCN` and `NGCF`, followed by re-evaluation of the selected best configuration with `seed ∈ {0,1,2,3,4}`. The `GraphSAGE` baseline row reported in the paper table is reused from the five-seed baseline evaluation.

Aggregated value tables and diagnostic definitions: `experiment_results/backbone.md`.

### Fusion-gate grid

**Branch:** `fusion-gate`

| Hyperparameter             | Values                                                                                                               |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `fusion_gate_version`      | `v1`, `v2`, `v3`                                                                                                     |
| `lambda_entropy`           | `1e-5`                                                                                                               |
| `gate_temperature`         | `1.0` (used only when `fusion_gate_version=v3`)                                                                      |
| `fixed_gates`              | `False` for learned-gate training; `True` for fixed-gate inference-only tests                                        |
| `fixed_gates_weights_name` | Filename of the best checkpoint selected from the corresponding learned-gate run (used only when `fixed_gates=True`) |

For fixed-gate inference-only tests, `fixed_gates_weights_name` expects only the checkpoint filename, not a full path. The corresponding file must be present under Elliot's weight directory, `results/weights/`, for example `results/weights/<fixed_gates_weights_name>`.

The fixed-gate checkpoints are provided in [`best_weights.zip`](https://github.com/doubleblind-artifact-2026/contextgnn-topk-artifact/releases/download/v1.0.0/best-weights.zip), available from the GitHub release assets. Extract the archive and place the checkpoint files in `results/weights/` before running the fixed-gate inference-only tests.

The fixed-gate runs are inference-only evaluations over saved learned-gate checkpoints, not additional model-training runs. The branch evaluates the fixed gate values `0.25`, `0.50`, and `0.75` during those inference-only tests.

**Multi-seed protocol:** direct multi-seed learned-gate evaluation with `seed ∈ {0,1,2,3,4}`; fixed-gate inference-only tests are then run on the corresponding five selected learned-gate checkpoints.

Aggregated learned-gate and fixed-gate value tables and diagnostic definitions: `experiment_results/fusion.md`.

### Contrastive-learning grid

**Branch:** `contrastive-learning`

| Method        | Hyperparameters                                                            |
| ------------- | -------------------------------------------------------------------------- |
| SGL-style     | `cl_method=sgl`, `lambda_cl ∈ {0.01, 0.05}`, `edge_drop_p ∈ {0.05, 0.20}`  |
| XSimGCL-style | `cl_method=xsimgcl`, `lambda_cl ∈ {0.01, 0.05}`, `xsim_eps ∈ {0.05, 0.20}` |
| Common CL     | `tau_cl=0.2`, `cl_layer=2`, `cl_scope=user_item`, `cl_normalize=True`      |

**Multi-seed protocol:** seed-0 screening for SGL and XSimGCL, followed by re-evaluation of the selected best configuration with `seed ∈ {0,1,2,3,4}`.

Aggregated value tables and diagnostic definitions: `experiment_results/contrastive.md`.

### Hard negative mining grid

**Branch:** `hard-negatives`

| Training regime                   | Hyperparameters                                                                                             |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| BCE uniform                       | `loss=bce`, `negative_sampling_strategy=uniform`                                                            |
| Global model-aware hard negatives | `loss=bce`, `negative_sampling_strategy=global_model_aware`, `hard_ratio ∈ {0.25, 0.75}`, `warmup_epochs=5` |
| Local model-aware hard negatives  | `loss=bce`, `negative_sampling_strategy=local_model_aware`, `hard_ratio ∈ {0.25, 0.75}`, `warmup_epochs=5`  |

The branch uses internal mining pool limits `global_hard_pool_topk=50` and `local_hard_pool_topk=20`. If `hard_ratio_applied=0` at the selected best epoch, the validation criterion selected a checkpoint still in the warmup/uniform regime. This is treated as an experimental outcome rather than as a parser error; mining-specific aggregated diagnostics are reported as unavailable when at least one seed has an undefined value.

**Multi-seed protocol:** seed-0 screening for global model-aware and local model-aware negative sampling, followed by re-evaluation of the selected best configuration with `seed ∈ {0,1,2,3,4}`. The BCE-uniform reference is evaluated directly with `seed ∈ {0,1,2,3,4}`.

Aggregated value tables and diagnostic definitions: `experiment_results/negatives.md`.

## Experiment Logs and Result Aggregation

The plain-text experiment logs used for the results reported in the paper and in the README tables are stored in:

```bash
experiment_logs/
```

The directory is organized by experimental block and dataset, so that the raw logs corresponding to each group of experiments can be inspected separately.

The `experiment_logs/` accounts for **426 model-training runs** and **45 additional fixed-gate inference-only runs**. Since each trained checkpoint is evaluated in the three inference modes `Full`, `Global`, and `Local`, the logs correspond to **1,278 mode-specific evaluations** from trained models, plus **135 mode-specific fixed-gate mode evaluations**, for a total of **1,413 mode-specific result evaluations**.&#x20;

The repository also provides the script:

```bash
summarize_experiments.py
```

This script reads the `.txt` log files stored in `experiment_logs/` and aggregates both ranking metrics and diagnostic logs as mean and standard deviation over the seeds used for each final evaluation. The aggregated Markdown result files are stored separately in `experiment_results/`.

The script accepts the following experiment-block identifiers as input:

| Argument      | Block                                           |
| ------------- | ----------------------------------------------- |
| `baseline`    | Reproduced ContextGNN baseline                  |
| `backbone`    | GNN-backbone analysis                           |
| `structure`   | Neighbor-sampling and feature-ablation analysis |
| `fusion`      | Fusion-gate analysis                            |
| `contrastive` | Contrastive-learning analysis                   |
| `negatives`   | Hard negative mining analysis                   |

For example, to aggregate the logs for the reproduced ContextGNN baseline, run:

```bash
python summarize_experiments.py baseline
```

The command prints the aggregated results to standard output.

The script is designed to work with the specific log files provided in `experiment_logs/`; the `experiment_logs/` directory, the `experiment_results/` directory, and `summarize_experiments.py` are available only in the `main` branch.

## Optional: Running Experiments on Kaggle

For users who do not have access to a local machine or server with a suitable GPU setup, the repository also provides the notebook:

```bash
run_experiments_kaggle.ipynb
```

The notebook is intended as a practical way to run the experiments in a free Kaggle environment, while handling the Python and dependency versions required by the project. It is available only in the `main` branch.

Before running the notebook, users should:

- set the Kaggle accelerator to **GPU P100**;
- replace `<TODO: branch>` with the branch name;
- adapt the training configuration inside the notebook according to the dataset and experimental setting to be reproduced;
- replace `<TODO: dataset>` with the desired dataset name, for example `amazon-book`.

The notebook is also prepared to resume an interrupted Kaggle session from the latest saved checkpoint, provided that the model saving and restoration parameters in the training configuration are not changed.

## Contributors

Anonymous Author(s)
