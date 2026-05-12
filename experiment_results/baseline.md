# Baseline Results

**Branch:** `main`

## Parsed Seeds and Best Epochs

| Dataset | Parsed seeds | Best epochs |
|---|---|---|
| Amazon-Book | `[0,1,2,3,4]` | `[42,27,9,28,30]` |
| Yelp2018 | `[0,1,2,3,4]` | `[42,30,40,39,54]` |
| Gowalla | `[0,1,2,3,4]` | `[31,33,37,33,47]` |

## Ranking Metrics on Test Set

| Dataset | Recall@20 Full | Recall@20 Global | Recall@20 Local | nDCG@20 Full | nDCG@20 Global | nDCG@20 Local |
|---|---:|---:|---:|---:|---:|---:|
| Amazon-Book | 0.042720 ± 0.004702 | 0.012640 ± 0.003665 | 0.042874 ± 0.004972 | 0.035446 ± 0.003800 | 0.010184 ± 0.002932 | 0.035526 ± 0.004007 |
| Yelp2018 | 0.053506 ± 0.000656 | 0.016096 ± 0.002440 | 0.053646 ± 0.001061 | 0.042720 ± 0.001146 | 0.012988 ± 0.001863 | 0.042896 ± 0.001401 |
| Gowalla | 0.157784 ± 0.012947 | 0.021338 ± 0.002228 | 0.157616 ± 0.012902 | 0.113576 ± 0.018131 | 0.016310 ± 0.002332 | 0.113524 ± 0.017977 |

## Diagnostics

| Dataset | Local top-K share | Global top-K share | Jaccard@20(Full, Local) | Jaccard@20(Full, Global) | Positive coverage | UB@20 | Zero positive coverage ratio | override_ratio | Local items/user | delta_override_mean | delta_override_std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Amazon-Book | 0.966600 ± 0.028895 | 0.033400 ± 0.028895 | 0.103680 ± 0.018802 | 0.047260 ± 0.020272 | 0.221800 ± 0.000000 | 0.276400 ± 0.000000 | 0.174400 ± 0.000000 | 0.016345 ± 0.000002 | 1497.194000 ± 0.177144 | 1.551840 ± 0.141863 | 1.937440 ± 0.141017 |
| Yelp2018 | 0.998820 ± 0.001424 | 0.001180 ± 0.001424 | 0.204420 ± 0.007197 | 0.031200 ± 0.007649 | 0.574100 ± 0.000000 | 0.633300 ± 0.000000 | 0.013800 ± 0.000000 | 0.059189 ± 0.000004 | 2252.012000 ± 0.141669 | 1.049280 ± 0.045319 | 2.006980 ± 0.198628 |
| Gowalla | 0.998120 ± 0.001574 | 0.001880 ± 0.001574 | 0.367520 ± 0.008691 | 0.021020 ± 0.003467 | 0.629100 ± 0.000000 | 0.687400 ± 0.000000 | 0.045300 ± 0.000000 | 0.036726 ± 0.000011 | 1505.066000 ± 0.442640 | 1.666800 ± 0.146091 | 2.577420 ± 0.124344 |

## Diagnostic Log Definitions

- `Full` inference uses the hybrid ContextGNN scoring used for the main recommendation output. In selective-override branches, local pair-wise scores replace global two-tower scores for locally sampled items; in the fusion-gate branch, local and global scores are blended by the learned or fixed gate.
- `Global` inference uses only the global two-tower score for all candidate items.
- `Local` inference uses only the local contextual pair-wise score for items present in the sampled local subgraph; non-local items are masked out.
- `Local top-K share` is the fraction of the generated top-K recommendations whose item belongs to the local sampled subgraph for the user.
- `Global top-K share` is the complement of `Local top-K share`, i.e., the fraction of top-K recommendations coming from items outside the local sampled subgraph.
- `Jaccard@20(Full, Local)` is the mean Jaccard overlap between the top-20 item sets produced by `Full` and `Local` inference for the same users.
- `Jaccard@20(Full, Global)` is the mean Jaccard overlap between the top-20 item sets produced by `Full` and `Global` inference for the same users.
- `Local items/user` (`local_items_per_user_mean`) is the average number of unique local candidate items sampled for each user in the evaluation batch.
- `Positive coverage` (`pos_coverage`) is the fraction of validation/test positive items that are present in the sampled local candidate set.
- `UB@20` is the local-candidate oracle upper bound on Recall@20: it measures how many positives could be recovered at most if the local candidate ranking were perfect.
- `Zero positive coverage ratio` (`users_with_zero_pos_coverage_ratio`) is the fraction of evaluated users with at least one positive item but no positive item in the sampled local candidate set.
- `override_ratio` is the fraction of the user-item score matrix affected by the local branch, computed as `num_overrides / (batch_size * num_items)`.
- `delta_override_mean` is the mean difference, over local candidates, between the local pair-wise score and the pre-override global two-tower score.
- `delta_override_std` is the standard deviation of the same local-minus-global score difference.
