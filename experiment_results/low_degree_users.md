# Low-Degree User Results

**Branch:** `main`

## Parsed Seeds

| Dataset | Parsed seeds |
|---|---|
| Amazon-Book | `[0,1,2,3,4]` |
| Yelp2018 | `[0,1,2,3,4]` |
| Gowalla | `[0,1,2,3,4]` |

## Degree Stratification

| Dataset | Total users | q25 | q75 |
|---|---:|---:|---:|
| Amazon-Book | 52639.000000 ± 0.000000 | 18.000000 ± 0.000000 | 44.000000 ± 0.000000 |
| Yelp2018 | 31668.000000 ± 0.000000 | 18.000000 ± 0.000000 | 40.000000 ± 0.000000 |
| Gowalla | 29858.000000 ± 0.000000 | 10.000000 ± 0.000000 | 28.000000 ± 0.000000 |

## Degree Groups

| Dataset | Group | Users | Degree min | Degree mean | Degree max |
|---|---|---:|---:|---:|---:|
| Amazon-Book | low | 13405.000000 ± 0.000000 | 15.000000 ± 0.000000 | 16.020000 ± 0.000000 | 18.000000 ± 0.000000 |
| Amazon-Book | medium | 26085.000000 ± 0.000000 | 19.000000 ± 0.000000 | 27.290000 ± 0.000000 | 44.000000 ± 0.000000 |
| Amazon-Book | high | 13149.000000 ± 0.000000 | 45.000000 ± 0.000000 | 106.570000 ± 0.000000 | 10681.000000 ± 0.000000 |
| Yelp2018 | low | 8466.000000 ± 0.000000 | 15.000000 ± 0.000000 | 16.040000 ± 0.000000 | 18.000000 ± 0.000000 |
| Yelp2018 | medium | 15360.000000 ± 0.000000 | 19.000000 ± 0.000000 | 26.170000 ± 0.000000 | 40.000000 ± 0.000000 |
| Yelp2018 | high | 7842.000000 ± 0.000000 | 41.000000 ± 0.000000 | 85.170000 ± 0.000000 | 1847.000000 ± 0.000000 |
| Gowalla | low | 8819.000000 ± 0.000000 | 7.000000 ± 0.000000 | 8.040000 ± 0.000000 | 10.000000 ± 0.000000 |
| Gowalla | medium | 13852.000000 ± 0.000000 | 11.000000 ± 0.000000 | 16.940000 ± 0.000000 | 28.000000 ± 0.000000 |
| Gowalla | high | 7187.000000 ± 0.000000 | 29.000000 ± 0.000000 | 66.040000 ± 0.000000 | 810.000000 ± 0.000000 |

## Ranking Metrics on Test Set

| Dataset | Degree group | Metric | Full | Global | Local |
|---|---|---|---:|---:|---:|
| Amazon-Book | low | Recall@20 | 0.058921 ± 0.005117 | 0.013682 ± 0.003762 | 0.058824 ± 0.005187 |
| Amazon-Book | low | nDCG@20 | 0.042347 ± 0.003878 | 0.008525 ± 0.002435 | 0.042241 ± 0.003642 |
| Amazon-Book | medium | Recall@20 | 0.045684 ± 0.003284 | 0.012610 ± 0.003585 | 0.045505 ± 0.003445 |
| Amazon-Book | medium | nDCG@20 | 0.036815 ± 0.002712 | 0.009302 ± 0.002580 | 0.036762 ± 0.002868 |
| Amazon-Book | high | Recall@20 | 0.020488 ± 0.001684 | 0.009586 ± 0.002378 | 0.020866 ± 0.001928 |
| Amazon-Book | high | nDCG@20 | 0.026112 ± 0.002494 | 0.012545 ± 0.002901 | 0.026588 ± 0.002663 |
| Yelp2018 | low | Recall@20 | 0.060499 ± 0.001558 | 0.016005 ± 0.002425 | 0.060824 ± 0.000467 |
| Yelp2018 | low | nDCG@20 | 0.039406 ± 0.001416 | 0.009658 ± 0.001541 | 0.039297 ± 0.000819 |
| Yelp2018 | medium | Recall@20 | 0.056146 ± 0.001021 | 0.015853 ± 0.002960 | 0.055871 ± 0.000926 |
| Yelp2018 | medium | nDCG@20 | 0.042322 ± 0.000990 | 0.011316 ± 0.002245 | 0.042140 ± 0.000837 |
| Yelp2018 | high | Recall@20 | 0.040921 ± 0.001268 | 0.014626 ± 0.002116 | 0.041511 ± 0.000778 |
| Yelp2018 | high | nDCG@20 | 0.047955 ± 0.001499 | 0.017498 ± 0.002273 | 0.049007 ± 0.001099 |
| Gowalla | low | Recall@20 | 0.190146 ± 0.008159 | 0.017096 ± 0.004010 | 0.190607 ± 0.007450 |
| Gowalla | low | nDCG@20 | 0.114419 ± 0.007327 | 0.010064 ± 0.002704 | 0.114555 ± 0.006821 |
| Gowalla | medium | Recall@20 | 0.172969 ± 0.005545 | 0.019797 ± 0.004078 | 0.172696 ± 0.005314 |
| Gowalla | medium | nDCG@20 | 0.126709 ± 0.006700 | 0.014025 ± 0.002950 | 0.126385 ± 0.006618 |
| Gowalla | high | Recall@20 | 0.121863 ± 0.003720 | 0.020637 ± 0.003350 | 0.122589 ± 0.003627 |
| Gowalla | high | nDCG@20 | 0.133198 ± 0.006862 | 0.024904 ± 0.004128 | 0.135159 ± 0.006704 |

## Diagnostic Log Definitions

- `Full` inference uses the hybrid ContextGNN scoring used for the main recommendation output. In selective-override branches, local pair-wise scores replace global two-tower scores for locally sampled items; in the fusion-gate branch, local and global scores are blended by the learned or fixed gate.
- `Global` inference uses only the global two-tower score for all candidate items.
- `Local` inference uses only the local contextual pair-wise score for items present in the sampled local subgraph; non-local items are masked out.
- `q25` and `q75` are the first and third quartile degree thresholds used to define the three user-degree groups.
- `low` contains users whose degree is at or below `q25`; `medium` contains users above `q25` and at or below `q75`; `high` contains users above `q75`.
- `Degree min`, `Degree mean`, and `Degree max` summarize the user degrees within each group.
- Ranking metrics are evaluated at K=20 separately for each degree group and inference mode.