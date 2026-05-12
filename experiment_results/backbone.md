# GNN Backbone Results

**Branch:** `main`

## Selected Configurations

| Dataset | GNN | Selected config | Seed-0 validation Recall@20 | Parsed seeds | Best epochs |
|---|---|---|---:|---|---|
| Amazon-Book | LightGCN | `lr=0.0005`, `n_layers=2`, `neigh=(16,16)` | 0.079630 | `[0,1,2,3,4]` | `[202,148,194,189,160]` |
| Amazon-Book | NGCF | `lr=0.005`, `n_layers=4`, `neigh=(16,16,8,8)` | 0.106980 | `[0,1,2,3,4]` | `[12,13,14,14,15]` |
| Yelp2018 | LightGCN | `lr=0.001`, `n_layers=2`, `neigh=(16,16)` | 0.075190 | `[0,1,2,3,4]` | `[78,153,89,69,123]` |
| Yelp2018 | NGCF | `lr=0.0005`, `n_layers=4`, `neigh=(16,16,16,16)` | 0.073700 | `[0,1,2,3,4]` | `[25,34,44,36,28]` |
| Gowalla | LightGCN | `lr=0.005`, `n_layers=3`, `neigh=(16,16,16)` | 0.163640 | `[0,1,2,3,4]` | `[1,1,2,3,1]` |
| Gowalla | NGCF | `lr=0.0005`, `n_layers=4`, `neigh=(16,16,16,16)` | 0.195120 | `[0,1,2,3,4]` | `[40,42,30,36,43]` |

## Ranking Metrics on Test Set

| Dataset | GNN | Recall@20 Full | Recall@20 Global | Recall@20 Local | nDCG@20 Full | nDCG@20 Global | nDCG@20 Local |
|---|---|---:|---:|---:|---:|---:|---:|
| Amazon-Book | LightGCN | 0.030912 ± 0.000978 | 0.030920 ± 0.001040 | 0.001080 ± 0.000000 | 0.024178 ± 0.000721 | 0.024154 ± 0.000800 | 0.000900 ± 0.000000 |
| Amazon-Book | NGCF | 0.040648 ± 0.000951 | 0.007678 ± 0.000383 | 0.041012 ± 0.001085 | 0.033388 ± 0.001001 | 0.006304 ± 0.000311 | 0.033690 ± 0.001022 |
| Yelp2018 | LightGCN | 0.049932 ± 0.002070 | 0.049858 ± 0.002408 | 0.003480 ± 0.000000 | 0.040106 ± 0.001795 | 0.039878 ± 0.001933 | 0.002030 ± 0.000000 |
| Yelp2018 | NGCF | 0.045520 ± 0.002118 | 0.006068 ± 0.001707 | 0.048078 ± 0.000675 | 0.035766 ± 0.002533 | 0.005096 ± 0.001645 | 0.039170 ± 0.000296 |
| Gowalla | LightGCN | 0.133744 ± 0.011382 | 0.001832 ± 0.001085 | 0.137004 ± 0.004907 | 0.108836 ± 0.011431 | 0.001450 ± 0.000971 | 0.112072 ± 0.005202 |
| Gowalla | NGCF | 0.162110 ± 0.001743 | 0.006940 ± 0.002088 | 0.162690 ± 0.001604 | 0.124006 ± 0.003924 | 0.005298 ± 0.001536 | 0.124452 ± 0.003697 |

## Diagnostics

| Dataset | GNN | Local top-K share | Global top-K share | Jaccard@20(Full, Local) | Jaccard@20(Full, Global) |
|---|---|---:|---:|---:|---:|
| Amazon-Book | LightGCN | 0.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.002040 ± 0.000410 | 0.425380 ± 0.003659 |
| Amazon-Book | NGCF | 0.974920 ± 0.026903 | 0.025080 ± 0.026903 | 0.090340 ± 0.004763 | 0.031080 ± 0.014067 |
| Yelp2018 | LightGCN | 0.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.006540 ± 0.001041 | 0.416780 ± 0.006613 |
| Yelp2018 | NGCF | 0.919880 ± 0.072916 | 0.080120 ± 0.072916 | 0.183720 ± 0.006494 | 0.032280 ± 0.019331 |
| Gowalla | LightGCN | 0.935980 ± 0.137163 | 0.064020 ± 0.137163 | 0.257900 ± 0.029539 | 0.042320 ± 0.089400 |
| Gowalla | NGCF | 0.994380 ± 0.006720 | 0.005620 ± 0.006720 | 0.368840 ± 0.011213 | 0.008060 ± 0.003120 |

## Diagnostic Log Definitions

- `Full` inference uses the hybrid ContextGNN scoring used for the main recommendation output. In selective-override branches, local pair-wise scores replace global two-tower scores for locally sampled items; in the fusion-gate branch, local and global scores are blended by the learned or fixed gate.
- `Global` inference uses only the global two-tower score for all candidate items.
- `Local` inference uses only the local contextual pair-wise score for items present in the sampled local subgraph; non-local items are masked out.
- `Local top-K share` is the fraction of the generated top-K recommendations whose item belongs to the local sampled subgraph for the user.
- `Global top-K share` is the complement of `Local top-K share`, i.e., the fraction of top-K recommendations coming from items outside the local sampled subgraph.
- `Jaccard@20(Full, Local)` is the mean Jaccard overlap between the top-20 item sets produced by `Full` and `Local` inference for the same users.
- `Jaccard@20(Full, Global)` is the mean Jaccard overlap between the top-20 item sets produced by `Full` and `Global` inference for the same users.
