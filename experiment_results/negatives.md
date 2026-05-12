# Hard Negative Mining Results

**Branch:** `hard-negatives`

When `hard_ratio_applied=0` at the best epoch, validation selected a checkpoint still in the warmup/uniform regime; this is an experimental result, not a parser error. Mining-specific diagnostics are reported as `--` when at least one seed has an undefined value, because averaging only the non-`nan` seeds would silently change the aggregation protocol.

## Selected Configurations

| Dataset | Strategy | Selected hard_ratio | Seed-0 validation Recall@20 | Parsed seeds | Best epochs |
|---|---|---:|---:|---|---|
| Amazon-Book | uniform | -- | -- | `[0,1,2,3,4]` | `[19,20,24,21,28]` |
| Amazon-Book | global_model_aware | 0.75 | 0.089390 | `[0,1,2,3,4]` | `[5,5,4,5,5]` |
| Amazon-Book | local_model_aware | 0.25 | 0.088600 | `[0,1,2,3,4]` | `[4,5,4,5,5]` |
| Yelp2018 | uniform | -- | -- | `[0,1,2,3,4]` | `[35,6,16,33,37]` |
| Yelp2018 | global_model_aware | 0.75 | 0.074270 | `[0,1,2,3,4]` | `[42,2,5,31,4]` |
| Yelp2018 | local_model_aware | 0.25 | 0.060910 | `[0,1,2,3,4]` | `[5,2,5,4,4]` |
| Gowalla | uniform | -- | -- | `[0,1,2,3,4]` | `[41,64,14,31,47]` |
| Gowalla | global_model_aware | 0.25 | 0.155340 | `[0,1,2,3,4]` | `[26,26,34,17,5]` |
| Gowalla | local_model_aware | 0.75 | 0.124220 | `[0,1,2,3,4]` | `[3,4,2,5,5]` |

## Ranking Metrics on Test Set

| Dataset | Strategy | Recall@20 Full | Recall@20 Global | Recall@20 Local | nDCG@20 Full | nDCG@20 Global | nDCG@20 Local |
|---|---|---:|---:|---:|---:|---:|---:|
| Amazon-Book | uniform | 0.039596 ± 0.000559 | 0.007170 ± 0.000243 | 0.041166 ± 0.000435 | 0.032766 ± 0.000500 | 0.005572 ± 0.000264 | 0.034298 ± 0.000354 |
| Amazon-Book | global_model_aware | 0.036652 ± 0.001627 | 0.002400 ± 0.000325 | 0.036330 ± 0.001284 | 0.030178 ± 0.001515 | 0.001938 ± 0.000370 | 0.029818 ± 0.001128 |
| Amazon-Book | local_model_aware | 0.034788 ± 0.001422 | 0.003634 ± 0.000629 | 0.034734 ± 0.001606 | 0.028600 ± 0.001391 | 0.002822 ± 0.000503 | 0.028518 ± 0.001554 |
| Yelp2018 | uniform | 0.045758 ± 0.002350 | 0.008992 ± 0.003520 | 0.046748 ± 0.002700 | 0.036328 ± 0.001927 | 0.007266 ± 0.002620 | 0.037196 ± 0.002286 |
| Yelp2018 | global_model_aware | 0.045580 ± 0.003842 | 0.000990 ± 0.000320 | 0.045364 ± 0.004010 | 0.036422 ± 0.003324 | 0.000740 ± 0.000225 | 0.036244 ± 0.003519 |
| Yelp2018 | local_model_aware | 0.042922 ± 0.001467 | 0.001702 ± 0.000304 | 0.042836 ± 0.001353 | 0.034262 ± 0.001796 | 0.001360 ± 0.000300 | 0.034168 ± 0.001547 |
| Gowalla | uniform | 0.146680 ± 0.007621 | 0.019514 ± 0.006747 | 0.148088 ± 0.008470 | 0.108320 ± 0.005977 | 0.014236 ± 0.004761 | 0.110100 ± 0.006224 |
| Gowalla | global_model_aware | 0.121240 ± 0.002821 | 0.004764 ± 0.003339 | 0.122346 ± 0.002643 | 0.085592 ± 0.005681 | 0.003110 ± 0.002236 | 0.086342 ± 0.005729 |
| Gowalla | local_model_aware | 0.125158 ± 0.009286 | 0.001356 ± 0.000966 | 0.125492 ± 0.009930 | 0.103156 ± 0.008972 | 0.000798 ± 0.000430 | 0.103246 ± 0.009394 |

## Loss and Score Diagnostics

| Dataset | Strategy | loss_pos | loss_neg | loss_total | Selected neg. score mean | selected_neg_score_std | pos_neg_margin_mean | pos_neg_margin_std |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Amazon-Book | uniform | 0.272592 ± 0.012913 | 0.266904 ± 0.007593 | 0.269748 ± 0.010248 | -3.372040 ± 0.275430 | 3.118360 ± 0.214298 | 9.296440 ± 0.476041 | 7.109080 ± 0.423675 |
| Amazon-Book | global_model_aware | 0.479324 ± 0.011551 | 0.365136 ± 0.005548 | 0.422232 ± 0.008529 | -1.092180 ± 0.053822 | 0.879420 ± 0.065269 | 6.458400 ± 0.387128 | 6.880460 ± 0.572406 |
| Amazon-Book | local_model_aware | 0.483092 ± 0.011495 | 0.366792 ± 0.005850 | 0.424942 ± 0.008667 | -1.074040 ± 0.061409 | 0.856600 ± 0.075398 | 6.423800 ± 0.337862 | 6.887340 ± 0.364068 |
| Yelp2018 | uniform | 0.230164 ± 0.079030 | 0.224954 ± 0.028636 | 0.227560 ± 0.053829 | -3.654840 ± 1.205105 | 2.608300 ± 0.874399 | 9.841900 ± 1.712648 | 6.697540 ± 0.496009 |
| Yelp2018 | global_model_aware | 0.381878 ± 0.003501 | 0.284492 ± 0.007862 | 0.333186 ± 0.005079 | -1.651760 ± 0.091120 | 1.243740 ± 0.283515 | 6.839600 ± 0.623555 | 5.894480 ± 0.755197 |
| Yelp2018 | local_model_aware | 0.380492 ± 0.003160 | 0.277924 ± 0.003083 | 0.329206 ± 0.003102 | -1.607880 ± 0.030622 | 1.059240 ± 0.030561 | 6.665800 ± 0.497144 | 5.700560 ± 0.654893 |
| Gowalla | uniform | 0.189642 ± 0.048644 | 0.171416 ± 0.019476 | 0.180530 ± 0.034055 | -4.786560 ± 1.303582 | 3.906100 ± 1.026099 | 11.482520 ± 1.986150 | 7.871080 ± 1.223737 |
| Gowalla | global_model_aware | 0.299900 ± 0.015263 | 0.226338 ± 0.003938 | 0.263120 ± 0.006003 | -2.831320 ± 0.365587 | 2.736480 ± 0.508763 | 8.409940 ± 0.646694 | 6.565920 ± 0.654614 |
| Gowalla | local_model_aware | 0.317412 ± 0.004475 | 0.222890 ± 0.003728 | 0.270150 ± 0.004061 | -2.438660 ± 0.026664 | 2.123340 ± 0.093962 | 7.885180 ± 0.456814 | 6.290860 ± 0.547753 |

## Sampling and Branch-Usage Diagnostics

| Dataset | Strategy | hard_ratio_applied | fallback_uniform_ratio | users_without_hard_candidates_ratio | Selected neg. pop. percentile | duplicate_neg_ratio | Local top-K share | Global top-K share | Jaccard@20(Full, Local) | Jaccard@20(Full, Global) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Amazon-Book | uniform | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.499820 ± 0.000164 | 0.004220 ± 0.000045 | 0.858060 ± 0.040766 | 0.141940 ± 0.040766 | 0.097920 ± 0.002923 | 0.091820 ± 0.016292 |
| Amazon-Book | global_model_aware | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.499620 ± 0.000349 | 0.004220 ± 0.000084 | 0.999980 ± 0.000045 | 0.000020 ± 0.000045 | 0.092880 ± 0.005607 | 0.003840 ± 0.001286 |
| Amazon-Book | local_model_aware | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.499500 ± 0.000374 | 0.004220 ± 0.000084 | 0.998200 ± 0.002263 | 0.001800 ± 0.002263 | 0.084520 ± 0.003593 | 0.007280 ± 0.002983 |
| Yelp2018 | uniform | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.499260 ± 0.000456 | 0.008420 ± 0.000110 | 0.961040 ± 0.031893 | 0.038960 ± 0.031893 | 0.191400 ± 0.014460 | 0.029920 ± 0.017536 |
| Yelp2018 | global_model_aware | 0.299860 ± 0.410600 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.550020 ± 0.069168 | 0.165440 ± 0.215436 | 1.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.185900 ± 0.015573 | 0.000680 ± 0.000618 |
| Yelp2018 | local_model_aware | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.499540 ± 0.000329 | 0.008300 ± 0.000071 | 1.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.177100 ± 0.004663 | 0.001060 ± 0.000723 |
| Gowalla | uniform | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.499980 ± 0.000576 | 0.005400 ± 0.000158 | 0.960120 ± 0.030873 | 0.039880 ± 0.030873 | 0.326940 ± 0.013749 | 0.031140 ± 0.018762 |
| Gowalla | global_model_aware | 0.199620 ± 0.111594 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.547680 ± 0.027949 | 0.037260 ± 0.018800 | 1.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.311900 ± 0.007196 | 0.000720 ± 0.000531 |
| Gowalla | local_model_aware | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.499460 ± 0.000483 | 0.005400 ± 0.000158 | 1.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.337080 ± 0.027331 | 0.000640 ± 0.000378 |

## Mining-Specific Diagnostics

Mining-specific diagnostics are not tabulated separately because every aggregated row would be unavailable under the current aggregation rule: at least one seed produces an undefined value for each metric row, and the aggregation deliberately does not drop such seeds.

## Diagnostic Log Definitions

- `Full` inference uses the hybrid ContextGNN scoring used for the main recommendation output. In selective-override branches, local pair-wise scores replace global two-tower scores for locally sampled items; in the fusion-gate branch, local and global scores are blended by the learned or fixed gate.
- `Global` inference uses only the global two-tower score for all candidate items.
- `Local` inference uses only the local contextual pair-wise score for items present in the sampled local subgraph; non-local items are masked out.
- `Local top-K share` is the fraction of the generated top-K recommendations whose item belongs to the local sampled subgraph for the user.
- `Global top-K share` is the complement of `Local top-K share`, i.e., the fraction of top-K recommendations coming from items outside the local sampled subgraph.
- `Jaccard@20(Full, Local)` is the mean Jaccard overlap between the top-20 item sets produced by `Full` and `Local` inference for the same users.
- `Jaccard@20(Full, Global)` is the mean Jaccard overlap between the top-20 item sets produced by `Full` and `Global` inference for the same users.
- `override_ratio` is the fraction of the user-item score matrix affected by the local branch, computed as `num_overrides / (batch_size * num_items)`.
- `delta_override_mean` is the mean difference, over local candidates, between the local pair-wise score and the pre-override global two-tower score.
- `delta_override_std` is the standard deviation of the same local-minus-global score difference.
- `loss_pos` is the BCE loss averaged on positive user-item pairs.
- `loss_neg` is the BCE loss averaged on sampled negative user-item pairs.
- `loss_total` is the balanced BCE objective, `0.5 * (loss_pos + loss_neg)`.
- `selected_neg_score_mean` and `selected_neg_score_std` summarize the model scores assigned to the selected negative items.
- `pos_neg_margin_mean` and `pos_neg_margin_std` summarize the score margin `positive_score - negative_score`.
- `hard_ratio_applied` is the realized fraction of supervised pairs for which a model-aware hard negative was successfully selected.
- `fallback_uniform_ratio` is the fraction of targeted hard-negative positions that fell back to uniform sampling because no valid hard candidate was available.
- `users_without_hard_candidates_ratio` is the fraction of users with no eligible hard-negative candidate under the selected mining strategy.
- `selected_neg_popularity_percentile_mean` is the mean training-set popularity percentile of the selected negative items.
- `duplicate_neg_ratio` is `1 - (#unique selected negatives / #selected negatives)` within the sampled negatives.
- `selected_neg_emb_score_mean` is the mean pre-override global two-tower score of negatives selected by `global_model_aware` mining.
- `selected_neg_outside_local_share` is the fraction of globally mined hard negatives that are outside the local sampled subgraph.
- `global_hard_pool_size_mean` is the mean per-user size of the global top-score hard-negative pool after masking seen items and truncating to `global_hard_pool_topk=50`.
- `local_hard_pool_size_mean` is the mean per-user number of eligible local unseen negatives before truncation to `local_hard_pool_topk=20`.
- `users_with_empty_local_pool_ratio` is the fraction of users with no eligible local hard-negative candidates.
- `selected_neg_local_rank_mean` is the mean local rank of the selected negative among eligible local hard candidates, where rank 1 is the highest-scoring local candidate.
- `--` marks diagnostics that are not available because at least one seed produced an undefined `nan` value; the aggregation deliberately does not drop such seeds.
