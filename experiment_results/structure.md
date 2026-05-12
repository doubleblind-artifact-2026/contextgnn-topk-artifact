# Structural Analysis Results

**Branch:** `structural-analysis`

This file reports the neighbor-sampling, Gowalla constant-encoding sanity-check, and feature-ablation results. Baseline-width rows in the neighbor-sampling tables reuse the shared baseline logs.

## Neighbor-Sampling Results

### Neighbor-Sampling Parsed Seeds and Best Epochs

| Dataset | `n_layers` | `neigh` | Parsed seeds | Best epochs |
|---|---:|---|---|---|
| Amazon-Book | 2 | `(4,4)` | `[0,1,2,3,4]` | `[119,131,141,146,117]` |
| Amazon-Book | 2 | `(8,8)` | `[0,1,2,3,4]` | `[108,137,134,116,138]` |
| Amazon-Book | 2 | `(16,16)` | `[0,1,2,3,4]` | `[81,92,78,86,87]` |
| Amazon-Book | 4 | `(4,4,4,4)` | `[0,1,2,3,4]` | `[165,207,180,177,159]` |
| Amazon-Book | 4 | `(8,8,8,8)` | `[0,1,2,3,4]` | `[53,87,109,64,97]` |
| Yelp2018 | 2 | `(4,4)` | `[0,1,2,3,4]` | `[81,72,99,77,79]` |
| Yelp2018 | 2 | `(8,8)` | `[0,1,2,3,4]` | `[73,80,74,66,81]` |
| Yelp2018 | 2 | `(16,16)` | `[0,1,2,3,4]` | `[72,59,48,67,42]` |
| Yelp2018 | 4 | `(4,4,4,4)` | `[0,1,2,3,4]` | `[166,140,157,114,110]` |
| Yelp2018 | 4 | `(8,8,8,8)` | `[0,1,2,3,4]` | `[64,9,92,24,61]` |
| Gowalla | 2 | `(4,4)` | `[0,1,2,3,4]` | `[87,63,63,102,100]` |
| Gowalla | 2 | `(8,8)` | `[0,1,2,3,4]` | `[76,64,51,54,55]` |
| Gowalla | 2 | `(16,16)` | `[0,1,2,3,4]` | `[51,58,50,62,46]` |
| Gowalla | 4 | `(4,4,4,4)` | `[0,1,2,3,4]` | `[104,144,132,65,108]` |
| Gowalla | 4 | `(8,8,8,8)` | `[0,1,2,3,4]` | `[47,49,31,49,28]` |

### Neighbor-Sampling Ranking Metrics

| Dataset | `n_layers` | `neigh` | Recall@20 Full | Recall@20 Global | Recall@20 Local | nDCG@20 Full | nDCG@20 Global | nDCG@20 Local |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Amazon-Book | 2 | `(4,4)` | 0.026574 ± 0.000511 | 0.026580 ± 0.000472 | 0.001080 ± 0.000000 | 0.020886 ± 0.000476 | 0.020918 ± 0.000471 | 0.000900 ± 0.000000 |
| Amazon-Book | 2 | `(8,8)` | 0.029534 ± 0.000688 | 0.029482 ± 0.000629 | 0.001080 ± 0.000000 | 0.023018 ± 0.000528 | 0.022992 ± 0.000503 | 0.000900 ± 0.000000 |
| Amazon-Book | 2 | `(16,16)` | 0.026330 ± 0.000752 | 0.026312 ± 0.000704 | 0.001080 ± 0.000000 | 0.021032 ± 0.000594 | 0.021054 ± 0.000647 | 0.000900 ± 0.000000 |
| Amazon-Book | 4 | `(4,4,4,4)` | 0.031612 ± 0.000936 | 0.031996 ± 0.000930 | 0.012848 ± 0.000275 | 0.024808 ± 0.000788 | 0.024936 ± 0.000898 | 0.011364 ± 0.000345 |
| Amazon-Book | 4 | `(8,8,8,8)` | 0.033750 ± 0.001202 | 0.025198 ± 0.001825 | 0.030726 ± 0.000641 | 0.027760 ± 0.000575 | 0.019850 ± 0.001416 | 0.025686 ± 0.000543 |
| Amazon-Book | 4 | `(16,16,8,8)` | 0.042720 ± 0.004702 | 0.012640 ± 0.003665 | 0.042874 ± 0.004972 | 0.035446 ± 0.003800 | 0.010184 ± 0.002932 | 0.035526 ± 0.004007 |
| Yelp2018 | 2 | `(4,4)` | 0.040524 ± 0.000338 | 0.040610 ± 0.000506 | 0.003480 ± 0.000000 | 0.032750 ± 0.000455 | 0.032868 ± 0.000566 | 0.002030 ± 0.000000 |
| Yelp2018 | 2 | `(8,8)` | 0.043544 ± 0.000672 | 0.043718 ± 0.000679 | 0.003480 ± 0.000000 | 0.034976 ± 0.000577 | 0.035142 ± 0.000510 | 0.002030 ± 0.000000 |
| Yelp2018 | 2 | `(16,16)` | 0.040718 ± 0.001972 | 0.040818 ± 0.001934 | 0.003480 ± 0.000000 | 0.033278 ± 0.001791 | 0.033308 ± 0.001791 | 0.002030 ± 0.000000 |
| Yelp2018 | 4 | `(4,4,4,4)` | 0.045952 ± 0.001201 | 0.048114 ± 0.001367 | 0.019770 ± 0.000469 | 0.037396 ± 0.000936 | 0.038986 ± 0.001069 | 0.016268 ± 0.000252 |
| Yelp2018 | 4 | `(8,8,8,8)` | 0.036186 ± 0.005439 | 0.035968 ± 0.009431 | 0.034386 ± 0.003522 | 0.029074 ± 0.004138 | 0.028962 ± 0.007607 | 0.027806 ± 0.002715 |
| Yelp2018 | 4 | `(16,16,16,16)` | 0.053506 ± 0.000656 | 0.016096 ± 0.002440 | 0.053646 ± 0.001061 | 0.042720 ± 0.001146 | 0.012988 ± 0.001863 | 0.042896 ± 0.001401 |
| Gowalla | 2 | `(4,4)` | 0.099338 ± 0.002127 | 0.099098 ± 0.002220 | 0.000910 ± 0.000000 | 0.079198 ± 0.003738 | 0.079012 ± 0.003865 | 0.000650 ± 0.000000 |
| Gowalla | 2 | `(8,8)` | 0.102038 ± 0.002853 | 0.102426 ± 0.002605 | 0.000910 ± 0.000000 | 0.085380 ± 0.002765 | 0.085498 ± 0.002655 | 0.000650 ± 0.000000 |
| Gowalla | 2 | `(16,16)` | 0.097620 ± 0.004789 | 0.097580 ± 0.004626 | 0.000910 ± 0.000000 | 0.080656 ± 0.004596 | 0.080702 ± 0.004674 | 0.000650 ± 0.000000 |
| Gowalla | 4 | `(4,4,4,4)` | 0.112580 ± 0.003585 | 0.118808 ± 0.005754 | 0.064502 ± 0.001149 | 0.088966 ± 0.003178 | 0.098464 ± 0.004471 | 0.047700 ± 0.000892 |
| Gowalla | 4 | `(8,8,8,8)` | 0.130048 ± 0.002330 | 0.073206 ± 0.007106 | 0.129292 ± 0.001932 | 0.098900 ± 0.002421 | 0.061066 ± 0.005903 | 0.098594 ± 0.002316 |
| Gowalla | 4 | `(16,16,16,16)` | 0.157784 ± 0.012947 | 0.021338 ± 0.002228 | 0.157616 ± 0.012902 | 0.113576 ± 0.018131 | 0.016310 ± 0.002332 | 0.113524 ± 0.017977 |

### Neighbor-Sampling Diagnostics

| Dataset | `n_layers` | `neigh` | Local top-K share | Global top-K share | Jaccard@20(Full, Local) | Jaccard@20(Full, Global) | Local items/user | Positive coverage | UB@20 | Zero positive coverage ratio |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Amazon-Book | 2 | `(4,4)` | 0.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.001840 ± 0.000270 | 0.891240 ± 0.010860 | 4.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 1.000000 ± 0.000000 |
| Amazon-Book | 2 | `(8,8)` | 0.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.001720 ± 0.000335 | 0.813500 ± 0.020702 | 8.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 1.000000 ± 0.000000 |
| Amazon-Book | 2 | `(16,16)` | 0.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.002460 ± 0.000371 | 0.720600 ± 0.014115 | 15.880000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 1.000000 ± 0.000000 |
| Amazon-Book | 4 | `(4,4,4,4)` | 0.043640 ± 0.016499 | 0.956360 ± 0.016499 | 0.020520 ± 0.000597 | 0.800660 ± 0.012201 | 63.120000 ± 0.018708 | 0.013460 ± 0.000089 | 0.016860 ± 0.000089 | 0.868420 ± 0.001148 |
| Amazon-Book | 4 | `(8,8,8,8)` | 0.508760 ± 0.046768 | 0.491240 ± 0.046768 | 0.070380 ± 0.005111 | 0.322980 ± 0.050156 | 453.184000 ± 0.077653 | 0.086380 ± 0.000217 | 0.108140 ± 0.000288 | 0.470320 ± 0.001969 |
| Amazon-Book | 4 | `(16,16,8,8)` | 0.966600 ± 0.028895 | 0.033400 ± 0.028895 | 0.103680 ± 0.018802 | 0.047260 ± 0.020272 | 1497.194000 ± 0.177144 | 0.221800 ± 0.000000 | 0.276400 ± 0.000000 | 0.174400 ± 0.000000 |
| Yelp2018 | 2 | `(4,4)` | 0.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.008000 ± 0.000797 | 0.826440 ± 0.018590 | 4.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 1.000000 ± 0.000000 |
| Yelp2018 | 2 | `(8,8)` | 0.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.007200 ± 0.000806 | 0.767200 ± 0.016979 | 8.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 1.000000 ± 0.000000 |
| Yelp2018 | 2 | `(16,16)` | 0.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.007480 ± 0.000817 | 0.726320 ± 0.016639 | 15.880000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 1.000000 ± 0.000000 |
| Yelp2018 | 4 | `(4,4,4,4)` | 0.013480 ± 0.003193 | 0.986520 ± 0.003193 | 0.032400 ± 0.001329 | 0.765020 ± 0.011138 | 63.124000 ± 0.019494 | 0.028240 ± 0.000434 | 0.032680 ± 0.000512 | 0.768940 ± 0.002731 |
| Yelp2018 | 4 | `(8,8,8,8)` | 0.620280 ± 0.224253 | 0.379720 ± 0.224253 | 0.072840 ± 0.017784 | 0.257340 ± 0.124710 | 435.806000 ± 0.056833 | 0.175600 ± 0.000837 | 0.203140 ± 0.000934 | 0.258800 ± 0.002021 |
| Yelp2018 | 4 | `(16,16,16,16)` | 0.998820 ± 0.001424 | 0.001180 ± 0.001424 | 0.204420 ± 0.007197 | 0.031200 ± 0.007649 | 2252.012000 ± 0.141669 | 0.574100 ± 0.000000 | 0.633300 ± 0.000000 | 0.013800 ± 0.000000 |
| Gowalla | 2 | `(4,4)` | 0.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.000940 ± 0.000182 | 0.763360 ± 0.048880 | 4.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 1.000000 ± 0.000000 |
| Gowalla | 2 | `(8,8)` | 0.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.001240 ± 0.000329 | 0.700420 ± 0.009904 | 7.860000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 1.000000 ± 0.000000 |
| Gowalla | 2 | `(16,16)` | 0.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.001760 ± 0.000754 | 0.753640 ± 0.011160 | 12.880000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 1.000000 ± 0.000000 |
| Gowalla | 4 | `(4,4,4,4)` | 0.231200 ± 0.070743 | 0.768800 ± 0.070743 | 0.054480 ± 0.001134 | 0.573800 ± 0.060110 | 59.410000 ± 0.027386 | 0.065660 ± 0.000564 | 0.074720 ± 0.000602 | 0.654840 ± 0.001903 |
| Gowalla | 4 | `(8,8,8,8)` | 0.881240 ± 0.036610 | 0.118760 ± 0.036610 | 0.160840 ± 0.008289 | 0.136740 ± 0.024290 | 377.578000 ± 0.127554 | 0.293780 ± 0.000792 | 0.334000 ± 0.000815 | 0.190320 ± 0.001057 |
| Gowalla | 4 | `(16,16,16,16)` | 0.998120 ± 0.001574 | 0.001880 ± 0.001574 | 0.367520 ± 0.008691 | 0.021020 ± 0.003467 | 1505.066000 ± 0.442640 | 0.629100 ± 0.000000 | 0.687400 ± 0.000000 | 0.045300 ± 0.000000 |

## Gowalla Constant-Encoding Sanity Check

This sanity check verifies that replacing the original constant-feature path with the unified dense constant-feature path preserves the qualitative operating regime. The wide setting remains Local-dominated, while the narrow 2-layer setting becomes Global-dominated once local positive coverage collapses.

The first two rows below reuse the corresponding Gowalla rows from the neighbor-sampling results, the third row reuses the Gowalla `const_unified` row from the feature-ablation results, and the fourth row is the additional narrow unified-constant sanity-check run.

### Sanity-Check Run Provenance

| Encoding path | Setting | Source row | Parsed seeds | Best epochs |
|---|---|---|---|---|
| Original baseline constant path | `n_layers=2`, `neigh=(16,16)` | Neighbor-sampling Gowalla (`n_layers=2`, `neigh=(16,16)`) | `[0,1,2,3,4]` | `[51,58,50,62,46]` |
| Original baseline constant path | `n_layers=4`, `neigh=(16,16,16,16)` | Neighbor-sampling Gowalla (`n_layers=4`, `neigh=(16,16,16,16)`) | reused shared baseline logs | reused shared baseline logs |
| Unified dense constant path | `n_layers=4`, `neigh=(16,16,16,16)` | Feature-ablation Gowalla `const_unified` | `[0,1,2,3,4]` | `[27,13,19,52,30]` |
| Unified dense constant path | `n_layers=2`, `neigh=(16,16)` | Additional sanity-check run | `[0,1,2]` | `[6,7,27]` |

### Sanity-Check Ranking Metrics

| Encoding path | Setting | Recall@20 Full | Recall@20 Global | Recall@20 Local | nDCG@20 Full | nDCG@20 Global | nDCG@20 Local |
|---|---|---:|---:|---:|---:|---:|---:|
| Original baseline constant path | `n_layers=2`, `neigh=(16,16)` | 0.097620 ± 0.004789 | 0.097580 ± 0.004626 | 0.000910 ± 0.000000 | 0.080656 ± 0.004596 | 0.080702 ± 0.004674 | 0.000650 ± 0.000000 |
| Original baseline constant path | `n_layers=4`, `neigh=(16,16,16,16)` | 0.157784 ± 0.012947 | 0.021338 ± 0.002228 | 0.157616 ± 0.012902 | 0.113576 ± 0.018131 | 0.016310 ± 0.002332 | 0.113524 ± 0.017977 |
| Unified dense constant path | `n_layers=4`, `neigh=(16,16,16,16)` | 0.164342 ± 0.002044 | 0.007130 ± 0.003147 | 0.165122 ± 0.001936 | 0.122942 ± 0.002281 | 0.005248 ± 0.002135 | 0.123206 ± 0.002246 |
| Unified dense constant path | `n_layers=2`, `neigh=(16,16)` | 0.033517 ± 0.003798 | 0.033613 ± 0.003789 | 0.000910 ± 0.000000 | 0.027310 ± 0.004019 | 0.027323 ± 0.003949 | 0.000650 ± 0.000000 |

### Sanity-Check Diagnostics

| Encoding path | Setting | Local top-K share | Global top-K share | Jaccard@20(Full, Local) | Jaccard@20(Full, Global) | Local items/user | Positive coverage | UB@20 | Zero positive coverage ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original baseline constant path | `n_layers=2`, `neigh=(16,16)` | 0.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.001760 ± 0.000754 | 0.753640 ± 0.011160 | 12.880000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 1.000000 ± 0.000000 |
| Original baseline constant path | `n_layers=4`, `neigh=(16,16,16,16)` | 0.998120 ± 0.001574 | 0.001880 ± 0.001574 | 0.367520 ± 0.008691 | 0.021020 ± 0.003467 | 1505.066000 ± 0.442640 | 0.629100 ± 0.000000 | 0.687400 ± 0.000000 | 0.045300 ± 0.000000 |
| Unified dense constant path | `n_layers=4`, `neigh=(16,16,16,16)` | 0.998440 ± 0.001592 | 0.001560 ± 0.001592 | 0.380580 ± 0.006246 | 0.004980 ± 0.002720 | 1505.036000 ± 0.297624 | 0.629340 ± 0.000590 | 0.687460 ± 0.000590 | 0.044460 ± 0.000488 |
| Unified dense constant path | `n_layers=2`, `neigh=(16,16)` | 0.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.931767 ± 0.042698 | 12.880000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 1.000000 ± 0.000000 |

## Feature-Ablation Results

### Feature-Ablation Parsed Seeds and Best Epochs

| Dataset | Feature type | Parsed seeds | Best epochs |
|---|---|---|---|
| Amazon-Book | `const_unified` | `[0,1,2,3,4]` | `[12,7,10,9,30]` |
| Amazon-Book | `random` | `[0,1,2,3,4]` | `[12,13,6,12,4]` |
| Amazon-Book | `informative` | `[0,1,2,3,4]` | `[26,33,28,30,17]` |
| Yelp2018 | `const_unified` | `[0,1,2,3,4]` | `[26,32,41,33,50]` |
| Yelp2018 | `random` | `[0,1,2,3,4]` | `[34,41,23,16,24]` |
| Yelp2018 | `informative` | `[0,1,2,3,4]` | `[10,14,22,16,14]` |
| Gowalla | `const_unified` | `[0,1,2,3,4]` | `[27,13,19,52,30]` |
| Gowalla | `random` | `[0,1,2,3,4]` | `[61,62,27,44,15]` |
| Gowalla | `informative` | `[0,1,2,3,4]` | `[15,9,7,10,17]` |

### Feature-Ablation Ranking Metrics

| Dataset | Feature type | Recall@20 Full | Recall@20 Global | Recall@20 Local | nDCG@20 Full | nDCG@20 Global | nDCG@20 Local |
|---|---|---:|---:|---:|---:|---:|---:|
| Amazon-Book | `const_unified` | 0.042458 ± 0.000510 | 0.005120 ± 0.001197 | 0.042552 ± 0.000756 | 0.035672 ± 0.000452 | 0.004172 ± 0.001012 | 0.035814 ± 0.000650 |
| Amazon-Book | `random` | 0.042000 ± 0.000664 | 0.005412 ± 0.000795 | 0.042070 ± 0.000721 | 0.035276 ± 0.000642 | 0.004488 ± 0.000654 | 0.035264 ± 0.000713 |
| Amazon-Book | `informative` | 0.043770 ± 0.000976 | 0.022604 ± 0.001420 | 0.043752 ± 0.001040 | 0.036166 ± 0.000835 | 0.017806 ± 0.001289 | 0.036116 ± 0.000964 |
| Yelp2018 | `const_unified` | 0.051562 ± 0.000658 | 0.007640 ± 0.001786 | 0.051136 ± 0.000690 | 0.041002 ± 0.000762 | 0.006160 ± 0.001328 | 0.040576 ± 0.001101 |
| Yelp2018 | `random` | 0.051078 ± 0.000839 | 0.012816 ± 0.002672 | 0.051272 ± 0.001005 | 0.040314 ± 0.001011 | 0.010402 ± 0.002013 | 0.040484 ± 0.001168 |
| Yelp2018 | `informative` | 0.052576 ± 0.000986 | 0.025382 ± 0.001267 | 0.052920 ± 0.000764 | 0.041766 ± 0.000807 | 0.020482 ± 0.000880 | 0.041868 ± 0.000788 |
| Gowalla | `const_unified` | 0.164342 ± 0.002044 | 0.007130 ± 0.003147 | 0.165122 ± 0.001936 | 0.122942 ± 0.002281 | 0.005248 ± 0.002135 | 0.123206 ± 0.002246 |
| Gowalla | `random` | 0.169136 ± 0.002734 | 0.018260 ± 0.009137 | 0.169070 ± 0.002945 | 0.128378 ± 0.003512 | 0.013692 ± 0.006762 | 0.128286 ± 0.003852 |
| Gowalla | `informative` | 0.158438 ± 0.005142 | 0.025598 ± 0.004455 | 0.158838 ± 0.005191 | 0.119514 ± 0.004783 | 0.018542 ± 0.003114 | 0.120294 ± 0.005233 |

### Feature-Ablation Diagnostics

| Dataset | Feature type | Local top-K share | Global top-K share | Jaccard@20(Full, Local) | Jaccard@20(Full, Global) | Local items/user | Positive coverage | UB@20 | Zero positive coverage ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Amazon-Book | `const_unified` | 0.999060 ± 0.001991 | 0.000940 ± 0.001991 | 0.100800 ± 0.002325 | 0.014600 ± 0.004655 | 1497.388000 ± 0.211353 | 0.222700 ± 0.000400 | 0.277600 ± 0.000515 | 0.172760 ± 0.001126 |
| Amazon-Book | `random` | 0.999820 ± 0.000217 | 0.000180 ± 0.000217 | 0.099820 ± 0.001335 | 0.017380 ± 0.004896 | 1497.276000 ± 0.163187 | 0.223060 ± 0.000434 | 0.277880 ± 0.000522 | 0.173780 ± 0.000746 |
| Amazon-Book | `informative` | 0.957240 ± 0.008014 | 0.042760 ± 0.008014 | 0.154060 ± 0.015007 | 0.119760 ± 0.010643 | 1497.226000 ± 0.242136 | 0.222660 ± 0.000297 | 0.277420 ± 0.000370 | 0.173540 ± 0.000844 |
| Yelp2018 | `const_unified` | 0.998820 ± 0.001612 | 0.001180 ± 0.001612 | 0.205360 ± 0.005922 | 0.006380 ± 0.003784 | 2252.254000 ± 0.126610 | 0.574000 ± 0.000911 | 0.633760 ± 0.001011 | 0.013540 ± 0.000658 |
| Yelp2018 | `random` | 0.995040 ± 0.006658 | 0.004960 ± 0.006658 | 0.207840 ± 0.004457 | 0.025100 ± 0.007409 | 2252.062000 ± 0.233495 | 0.573840 ± 0.000841 | 0.633460 ± 0.000691 | 0.013780 ± 0.000383 |
| Yelp2018 | `informative` | 0.990900 ± 0.004660 | 0.009100 ± 0.004660 | 0.244860 ± 0.006835 | 0.077720 ± 0.008310 | 2251.922000 ± 0.518479 | 0.574480 ± 0.000736 | 0.634140 ± 0.000677 | 0.013840 ± 0.000358 |
| Gowalla | `const_unified` | 0.998440 ± 0.001592 | 0.001560 ± 0.001592 | 0.380580 ± 0.006246 | 0.004980 ± 0.002720 | 1505.036000 ± 0.297624 | 0.629340 ± 0.000590 | 0.687460 ± 0.000590 | 0.044460 ± 0.000488 |
| Gowalla | `random` | 0.994400 ± 0.004848 | 0.005600 ± 0.004848 | 0.385320 ± 0.009244 | 0.021540 ± 0.010657 | 1504.960000 ± 0.083367 | 0.629520 ± 0.000349 | 0.687660 ± 0.000251 | 0.045340 ± 0.000750 |
| Gowalla | `informative` | 0.989180 ± 0.007850 | 0.010820 ± 0.007850 | 0.346800 ± 0.025735 | 0.028380 ± 0.005718 | 1505.026000 ± 0.220862 | 0.629560 ± 0.000871 | 0.687840 ± 0.000777 | 0.045000 ± 0.000768 |

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
