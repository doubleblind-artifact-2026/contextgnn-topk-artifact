# Fusion-Gate Results

**Branch:** `fusion-gate`

## Learned Gate Parsed Seeds and Best Epochs

| Dataset | Gate | Parsed seeds | Best epochs |
|---|---|---|---|
| Amazon-Book | `v1` | `[0,1,2,3,4]` | `[23,26,21,23,63]` |
| Amazon-Book | `v2` | `[0,1,2,3,4]` | `[31,59,61,65,15]` |
| Amazon-Book | `v3` | `[0,1,2,3,4]` | `[63,59,58,85,28]` |
| Yelp2018 | `v1` | `[0,1,2,3,4]` | `[20,27,37,53,29]` |
| Yelp2018 | `v2` | `[0,1,2,3,4]` | `[41,29,30,37,24]` |
| Yelp2018 | `v3` | `[0,1,2,3,4]` | `[42,17,31,38,24]` |
| Gowalla | `v1` | `[0,1,2,3,4]` | `[28,31,31,32,30]` |
| Gowalla | `v2` | `[0,1,2,3,4]` | `[35,19,29,32,40]` |
| Gowalla | `v3` | `[0,1,2,3,4]` | `[32,27,29,28,23]` |

## Learned Gate Ranking Metrics

| Dataset | Gate | Recall@20 Full | Recall@20 Global | Recall@20 Local | nDCG@20 Full | nDCG@20 Global | nDCG@20 Local |
|---|---|---:|---:|---:|---:|---:|---:|
| Amazon-Book | `v1` | 0.044576 ± 0.000216 | 0.014756 ± 0.002840 | 0.010290 ± 0.003633 | 0.036646 ± 0.000408 | 0.011934 ± 0.002012 | 0.006712 ± 0.002575 |
| Amazon-Book | `v2` | 0.045630 ± 0.001692 | 0.017434 ± 0.005175 | 0.026054 ± 0.006950 | 0.037544 ± 0.001313 | 0.013904 ± 0.004085 | 0.018044 ± 0.006035 |
| Amazon-Book | `v3` | 0.046106 ± 0.000492 | 0.020598 ± 0.002094 | 0.020630 ± 0.004260 | 0.037724 ± 0.000465 | 0.016486 ± 0.001604 | 0.014900 ± 0.004537 |
| Yelp2018 | `v1` | 0.053364 ± 0.000698 | 0.026204 ± 0.004301 | 0.021682 ± 0.004607 | 0.042682 ± 0.000620 | 0.021312 ± 0.003531 | 0.015816 ± 0.003051 |
| Yelp2018 | `v2` | 0.052566 ± 0.000926 | 0.024800 ± 0.002212 | 0.021456 ± 0.009143 | 0.041380 ± 0.000839 | 0.020128 ± 0.001867 | 0.015918 ± 0.006594 |
| Yelp2018 | `v3` | 0.052404 ± 0.001376 | 0.023552 ± 0.003211 | 0.030402 ± 0.006739 | 0.041542 ± 0.001472 | 0.019420 ± 0.002589 | 0.021788 ± 0.005287 |
| Gowalla | `v1` | 0.163790 ± 0.005101 | 0.046750 ± 0.003625 | 0.101822 ± 0.013087 | 0.120696 ± 0.007901 | 0.035768 ± 0.002662 | 0.063012 ± 0.009441 |
| Gowalla | `v2` | 0.161468 ± 0.014274 | 0.043318 ± 0.005313 | 0.132512 ± 0.026947 | 0.119336 ± 0.018009 | 0.033668 ± 0.004874 | 0.089062 ± 0.025930 |
| Gowalla | `v3` | 0.164004 ± 0.006942 | 0.039634 ± 0.008974 | 0.147832 ± 0.009061 | 0.121712 ± 0.009172 | 0.031436 ± 0.007345 | 0.102786 ± 0.010553 |

## Learned Gate Diagnostics: Gate Statistics

| Dataset | Gate | gate_mean | gate_std | gate_min | gate_max | gate_entropy_mean | gate_temperature | topk_local_gate_mean@20 | topk_local_with_gate_high_share@20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Amazon-Book | `v1` | 0.246980 ± 0.021203 | 0.150220 ± 0.006316 | 0.001880 ± 0.002701 | 1.000000 ± 0.000000 | 0.495600 ± 0.024454 | -- | 0.597360 ± 0.050612 | 0.748440 ± 0.119588 |
| Amazon-Book | `v2` | 0.247680 ± 0.045877 | 0.152160 ± 0.006004 | 0.003180 ± 0.006672 | 1.000000 ± 0.000000 | 0.491180 ± 0.057084 | -- | 0.578000 ± 0.061147 | 0.733500 ± 0.155573 |
| Amazon-Book | `v3` | 0.227940 ± 0.030494 | 0.158980 ± 0.005074 | 0.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.462560 ± 0.040074 | 0.769160 ± 0.155379 | 0.589680 ± 0.027161 | 0.741660 ± 0.093084 |
| Yelp2018 | `v1` | 0.409400 ± 0.065824 | 0.210520 ± 0.014028 | 0.015200 ± 0.027120 | 0.999920 ± 0.000179 | 0.557560 ± 0.036985 | -- | 0.716440 ± 0.032407 | 0.969400 ± 0.022426 |
| Yelp2018 | `v2` | 0.412720 ± 0.072588 | 0.198160 ± 0.032538 | 0.032640 ± 0.052157 | 1.000000 ± 0.000000 | 0.570820 ± 0.039117 | -- | 0.711960 ± 0.079855 | 0.942680 ± 0.059442 |
| Yelp2018 | `v3` | 0.457480 ± 0.091435 | 0.210980 ± 0.006951 | 0.065700 ± 0.100798 | 1.000000 ± 0.000000 | 0.565200 ± 0.028862 | 0.924600 ± 0.071717 | 0.750020 ± 0.015046 | 0.990680 ± 0.008305 |
| Gowalla | `v1` | 0.530040 ± 0.009486 | 0.178220 ± 0.009494 | 0.044520 ± 0.017527 | 0.999940 ± 0.000089 | 0.618200 ± 0.008094 | -- | 0.823120 ± 0.027077 | 0.998900 ± 0.001065 |
| Gowalla | `v2` | 0.590420 ± 0.117596 | 0.189460 ± 0.017168 | 0.108020 ± 0.072846 | 0.999920 ± 0.000084 | 0.563560 ± 0.045101 | -- | 0.902540 ± 0.071678 | 0.991540 ± 0.018638 |
| Gowalla | `v3` | 0.625960 ± 0.028129 | 0.193260 ± 0.015418 | 0.125620 ± 0.025130 | 0.999940 ± 0.000089 | 0.566820 ± 0.022167 | 0.993820 ± 0.017190 | 0.930680 ± 0.018706 | 0.999380 ± 0.001221 |

## Learned Gate Diagnostics: Branch Usage and Overrides

| Dataset | Gate | override_ratio | Local items/user | delta_override_mean | delta_override_std | Local top-K share | Global top-K share | Jaccard@20(Full, Local) | Jaccard@20(Full, Global) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Amazon-Book | `v1` | 0.016349 ± 0.000003 | 1497.514000 ± 0.234051 | 4.669520 ± 0.141560 | 3.691100 ± 0.164775 | 0.969300 ± 0.028890 | 0.030700 ± 0.028890 | 0.007040 ± 0.003632 | 0.113460 ± 0.052194 |
| Amazon-Book | `v2` | 0.016345 ± 0.000003 | 1497.212000 ± 0.310274 | 4.153100 ± 0.352648 | 3.404660 ± 0.369809 | 0.945740 ± 0.038222 | 0.054260 ± 0.038222 | 0.036980 ± 0.016976 | 0.137840 ± 0.060406 |
| Amazon-Book | `v3` | 0.016343 ± 0.000002 | 1497.010000 ± 0.195959 | 4.123340 ± 0.310436 | 3.666480 ± 0.292608 | 0.921420 ± 0.038481 | 0.078580 ± 0.038481 | 0.022160 ± 0.010422 | 0.168820 ± 0.050792 |
| Yelp2018 | `v1` | 0.059182 ± 0.000006 | 2251.740000 ± 0.226716 | 2.559780 ± 0.439940 | 3.155180 ± 0.549839 | 0.998060 ± 0.002495 | 0.001940 ± 0.002495 | 0.045460 ± 0.012393 | 0.136960 ± 0.039863 |
| Yelp2018 | `v2` | 0.059194 ± 0.000007 | 2252.202000 ± 0.277795 | 2.377040 ± 0.505018 | 3.011840 ± 0.519335 | 0.997140 ± 0.002373 | 0.002860 ± 0.002373 | 0.054680 ± 0.032725 | 0.120160 ± 0.010007 |
| Yelp2018 | `v3` | 0.059192 ± 0.000010 | 2252.146000 ± 0.385979 | 2.222800 ± 0.516138 | 2.690380 ± 0.609011 | 0.997880 ± 0.002867 | 0.002120 ± 0.002867 | 0.076620 ± 0.022161 | 0.116600 ± 0.022963 |
| Gowalla | `v1` | 0.036721 ± 0.000012 | 1504.872000 ± 0.477253 | 2.798340 ± 0.188785 | 3.594240 ± 0.102130 | 0.997180 ± 0.002022 | 0.002820 ± 0.002022 | 0.202580 ± 0.015540 | 0.083880 ± 0.009283 |
| Gowalla | `v2` | 0.036728 ± 0.000013 | 1505.164000 ± 0.521037 | 2.360320 ± 0.464696 | 3.261320 ± 0.464699 | 0.993720 ± 0.004142 | 0.006280 ± 0.004142 | 0.285600 ± 0.085328 | 0.071400 ± 0.009370 |
| Gowalla | `v3` | 0.036730 ± 0.000015 | 1505.206000 ± 0.625244 | 2.167220 ± 0.125054 | 3.123000 ± 0.097838 | 0.989100 ± 0.007184 | 0.010900 ± 0.007184 | 0.323960 ± 0.017542 | 0.068500 ± 0.002961 |

## Fixed-Gate Inference Ranking Metrics

| Dataset | Fixed `g` | Recall@20 Full | Recall@20 Global | Recall@20 Local | nDCG@20 Full | nDCG@20 Global | nDCG@20 Local |
|---|---:|---:|---:|---:|---:|---:|---:|
| Amazon-Book | 0.25 | 0.027740 ± 0.001952 | 0.019168 ± 0.002360 | 0.024194 ± 0.007258 | 0.021372 ± 0.001501 | 0.015308 ± 0.001863 | 0.017462 ± 0.005953 |
| Amazon-Book | 0.50 | 0.032630 ± 0.004687 | 0.019164 ± 0.002335 | 0.024116 ± 0.007380 | 0.024356 ± 0.004891 | 0.015296 ± 0.001858 | 0.017432 ± 0.006056 |
| Amazon-Book | 0.75 | 0.029220 ± 0.007375 | 0.019152 ± 0.002252 | 0.024172 ± 0.007425 | 0.021862 ± 0.007100 | 0.015316 ± 0.001838 | 0.017420 ± 0.005990 |
| Yelp2018 | 0.25 | 0.032924 ± 0.003583 | 0.025698 ± 0.004228 | 0.023900 ± 0.006810 | 0.026700 ± 0.003001 | 0.021130 ± 0.003657 | 0.017470 ± 0.004437 |
| Yelp2018 | 0.50 | 0.035190 ± 0.010861 | 0.025658 ± 0.004290 | 0.023888 ± 0.006829 | 0.027800 ± 0.009074 | 0.021106 ± 0.003635 | 0.017458 ± 0.004448 |
| Yelp2018 | 0.75 | 0.029638 ± 0.013421 | 0.025624 ± 0.004150 | 0.023790 ± 0.006722 | 0.022406 ± 0.010652 | 0.021072 ± 0.003614 | 0.017392 ± 0.004438 |
| Gowalla | 0.25 | 0.071626 ± 0.007657 | 0.028304 ± 0.005628 | 0.140270 ± 0.031947 | 0.051980 ± 0.004544 | 0.022172 ± 0.004449 | 0.096614 ± 0.028934 |
| Gowalla | 0.50 | 0.126502 ± 0.004302 | 0.028336 ± 0.005698 | 0.140228 ± 0.031818 | 0.092294 ± 0.002316 | 0.022178 ± 0.004462 | 0.096550 ± 0.028903 |
| Gowalla | 0.75 | 0.149898 ± 0.014654 | 0.028394 ± 0.005626 | 0.139926 ± 0.032371 | 0.107908 ± 0.016302 | 0.022210 ± 0.004435 | 0.096430 ± 0.029384 |

## Fixed-Gate Inference Diagnostics

| Dataset | Fixed `g` | topk_local_gate_mean@20 | topk_local_with_gate_high_share@20 | Local top-K share | Global top-K share | Jaccard@20(Full, Local) | Jaccard@20(Full, Global) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Amazon-Book | 0.25 | 0.250000 ± 0.000000 | 0.000000 ± 0.000000 | 0.813780 ± 0.046454 | 0.186220 ± 0.046454 | 0.011380 ± 0.007941 | 0.364580 ± 0.046993 |
| Amazon-Book | 0.50 | 0.500000 ± 0.000000 | 0.000000 ± 0.000000 | 0.948700 ± 0.023990 | 0.051300 ± 0.023990 | 0.031900 ± 0.013383 | 0.142620 ± 0.037816 |
| Amazon-Book | 0.75 | 0.750000 ± 0.000000 | 1.000000 ± 0.000000 | 0.990780 ± 0.005808 | 0.009220 ± 0.005808 | 0.050720 ± 0.012052 | 0.032280 ± 0.017577 |
| Yelp2018 | 0.25 | 0.250000 ± 0.000000 | 0.000000 ± 0.000000 | 0.954700 ± 0.009176 | 0.045300 ± 0.009176 | 0.010800 ± 0.010685 | 0.497760 ± 0.044750 |
| Yelp2018 | 0.50 | 0.500000 ± 0.000000 | 0.000000 ± 0.000000 | 0.984900 ± 0.010035 | 0.015100 ± 0.010035 | 0.034360 ± 0.013257 | 0.225120 ± 0.095072 |
| Yelp2018 | 0.75 | 0.750000 ± 0.000000 | 1.000000 ± 0.000000 | 0.995920 ± 0.003283 | 0.004080 ± 0.003283 | 0.089780 ± 0.005631 | 0.060500 ± 0.046718 |
| Gowalla | 0.25 | 0.250000 ± 0.000000 | 0.000000 ± 0.000000 | 0.863160 ± 0.055974 | 0.136840 ± 0.055974 | 0.052980 ± 0.024518 | 0.433420 ± 0.050226 |
| Gowalla | 0.50 | 0.500000 ± 0.000000 | 0.000000 ± 0.000000 | 0.971800 ± 0.015864 | 0.028200 ± 0.015864 | 0.144340 ± 0.043858 | 0.197880 ± 0.023810 |
| Gowalla | 0.75 | 0.750000 ± 0.000000 | 1.000000 ± 0.000000 | 0.990580 ± 0.008336 | 0.009420 ± 0.008336 | 0.277560 ± 0.032958 | 0.074640 ± 0.027539 |

## Diagnostic Log Definitions

- `Full` inference uses the hybrid ContextGNN scoring used for the main recommendation output. In selective-override branches, local pair-wise scores replace global two-tower scores for locally sampled items; in the fusion-gate branch, local and global scores are blended by the learned or fixed gate.
- `Global` inference uses only the global two-tower score for all candidate items.
- `Local` inference uses only the local contextual pair-wise score for items present in the sampled local subgraph; non-local items are masked out.
- `Local top-K share` is the fraction of the generated top-K recommendations whose item belongs to the local sampled subgraph for the user.
- `Global top-K share` is the complement of `Local top-K share`, i.e., the fraction of top-K recommendations coming from items outside the local sampled subgraph.
- `Jaccard@20(Full, Local)` is the mean Jaccard overlap between the top-20 item sets produced by `Full` and `Local` inference for the same users.
- `Jaccard@20(Full, Global)` is the mean Jaccard overlap between the top-20 item sets produced by `Full` and `Global` inference for the same users.
- `gate_mean` is the average learned gate value `g` over local user-item candidates. In `Full` inference, local candidates are scored as `g * s_local + (1 - g) * s_global`, where `s_local` is the local pair-wise score and `s_global` is the global two-tower score.
- `gate_std`, `gate_min`, and `gate_max` summarize the dispersion and range of the learned gate values over local candidates.
- `gate_entropy_mean` is the mean binary entropy of the learned gate values; lower values indicate more decisive gates close to 0 or 1.
- `gate_temperature` is the positive temperature used only by `fusion_gate_version=v3`, where the gate is computed as `sigmoid(gate_logit / temperature)`.
- `topk_local_gate_mean@20` is the mean gate value among top-20 recommended items that are also local candidates.
- `topk_local_with_gate_high_share@20` is the fraction of top-20 local recommendations whose gate value is greater than `0.5`.
- `override_ratio` is the fraction of the user-item score matrix affected by the local branch, computed as `num_overrides / (batch_size * num_items)`.
- `delta_override_mean` is the mean difference, over local candidates, between the local pair-wise score and the pre-override global two-tower score.
- `delta_override_std` is the standard deviation of the same local-minus-global score difference.
