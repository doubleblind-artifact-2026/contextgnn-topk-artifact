# Contrastive-Learning Results

**Branch:** `contrastive-learning`

## Selected Configurations

| Dataset | Method | Selected config | Seed-0 validation Recall@20 | Parsed seeds | Best epochs |
|---|---|---|---:|---|---|
| Amazon-Book | SGL | `cl_method=sgl`, `lambda_cl=0.01`, `tau_cl=0.2`, `edge_drop_p=0.20`, `xsim_eps=0.10` | 0.118080 | `[0,1,2,3,4]` | `[33,44,38,28,30]` |
| Amazon-Book | XSimGCL | `cl_method=xsimgcl`, `lambda_cl=0.05`, `tau_cl=0.2`, `edge_drop_p=0.10`, `xsim_eps=0.05` | 0.117570 | `[0,1,2,3,4]` | `[34,77,43,28,30]` |
| Yelp2018 | SGL | `cl_method=sgl`, `lambda_cl=0.05`, `tau_cl=0.2`, `edge_drop_p=0.05`, `xsim_eps=0.10` | 0.086360 | `[0,1,2,3,4]` | `[42,30,39,33,30]` |
| Yelp2018 | XSimGCL | `cl_method=xsimgcl`, `lambda_cl=0.01`, `tau_cl=0.2`, `edge_drop_p=0.10`, `xsim_eps=0.05` | 0.087980 | `[0,1,2,3,4]` | `[54,29,54,33,46]` |
| Gowalla | SGL | `cl_method=sgl`, `lambda_cl=0.05`, `tau_cl=0.2`, `edge_drop_p=0.20`, `xsim_eps=0.10` | 0.208420 | `[0,1,2,3,4]` | `[18,27,25,22,26]` |
| Gowalla | XSimGCL | `cl_method=xsimgcl`, `lambda_cl=0.01`, `tau_cl=0.2`, `edge_drop_p=0.10`, `xsim_eps=0.05` | 0.208720 | `[0,1,2,3,4]` | `[19,37,23,33,26]` |

## Ranking Metrics on Test Set

| Dataset | Method | Recall@20 Full | Recall@20 Global | Recall@20 Local | nDCG@20 Full | nDCG@20 Global | nDCG@20 Local |
|---|---|---:|---:|---:|---:|---:|---:|
| Amazon-Book | SGL | 0.044596 ± 0.000458 | 0.014650 ± 0.001183 | 0.044606 ± 0.000589 | 0.036970 ± 0.000348 | 0.011692 ± 0.000918 | 0.036922 ± 0.000479 |
| Amazon-Book | XSimGCL | 0.045338 ± 0.000751 | 0.016002 ± 0.001889 | 0.045564 ± 0.000569 | 0.037808 ± 0.000514 | 0.012894 ± 0.001515 | 0.038088 ± 0.000606 |
| Yelp2018 | SGL | 0.053830 ± 0.001019 | 0.015298 ± 0.001338 | 0.054226 ± 0.000745 | 0.042936 ± 0.000817 | 0.012480 ± 0.001092 | 0.043262 ± 0.000473 |
| Yelp2018 | XSimGCL | 0.053776 ± 0.000824 | 0.016506 ± 0.002648 | 0.054056 ± 0.000762 | 0.042964 ± 0.001019 | 0.013390 ± 0.002119 | 0.043360 ± 0.000884 |
| Gowalla | SGL | 0.168304 ± 0.006870 | 0.015882 ± 0.002121 | 0.168334 ± 0.007104 | 0.128442 ± 0.010452 | 0.011848 ± 0.001686 | 0.128496 ± 0.010403 |
| Gowalla | XSimGCL | 0.166220 ± 0.003205 | 0.018312 ± 0.003394 | 0.166156 ± 0.003049 | 0.125284 ± 0.005107 | 0.013782 ± 0.002536 | 0.125790 ± 0.005258 |

## Common Contrastive Diagnostics

| Dataset | Method | loss_sup | loss_cl | loss_total | cl_to_sup_ratio | pos_sim_mean | pos_sim_std | neg_sim_mean | neg_sim_std | hardest_neg_sim_mean | alignment_mean | uniformity_mean | user_emb_std_per_dim_mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Amazon-Book | SGL | 174.766308 ± 1.581151 | 3.897062 ± 0.094784 | 174.805276 ± 1.582053 | 0.000258 ± 0.000004 | 0.833560 ± 0.015876 | 0.177862 ± 0.014967 | 0.618552 ± 0.028875 | 0.197056 ± 0.010028 | 0.963810 ± 0.005428 | 0.332880 ± 0.031751 | -1.225520 ± 0.097671 | 0.070638 ± 0.002361 |
| Amazon-Book | XSimGCL | 174.055168 ± 3.416762 | 4.199094 ± 0.279323 | 174.265122 ± 3.430673 | 0.001406 ± 0.000068 | 0.594018 ± 0.034614 | 0.086468 ± 0.008701 | 0.362412 ± 0.033894 | 0.158966 ± 0.006466 | 0.636942 ± 0.029057 | 0.811962 ± 0.069226 | -1.415430 ± 0.254229 | 0.072692 ± 0.003737 |
| Yelp2018 | SGL | 124.220508 ± 0.591787 | 3.565792 ± 0.169181 | 124.398796 ± 0.598818 | 0.001548 ± 0.000064 | 0.993654 ± 0.000838 | 0.010904 ± 0.001884 | 0.604746 ± 0.035444 | 0.218876 ± 0.017572 | 0.979704 ± 0.004984 | 0.012696 ± 0.001674 | -1.236778 ± 0.125344 | 0.088740 ± 0.004287 |
| Yelp2018 | XSimGCL | 123.839984 ± 1.408978 | 4.834032 ± 0.032220 | 123.888324 ± 1.409190 | 0.000424 ± 0.000005 | 0.292278 ± 0.059314 | 0.046628 ± 0.011047 | 0.272548 ± 0.063999 | 0.047548 ± 0.010453 | 0.391406 ± 0.050596 | 1.415442 ± 0.118633 | -0.777188 ± 0.157801 | 0.082362 ± 0.005343 |
| Gowalla | SGL | 78.824328 ± 0.777670 | 3.599686 ± 0.099641 | 79.004312 ± 0.782487 | 0.002622 ± 0.000054 | 0.959764 ± 0.003713 | 0.053342 ± 0.006307 | 0.542376 ± 0.022432 | 0.253426 ± 0.007382 | 0.974526 ± 0.004005 | 0.080472 ± 0.007429 | -1.342702 ± 0.082427 | 0.107364 ± 0.003114 |
| Gowalla | XSimGCL | 78.225428 ± 1.165407 | 4.631302 ± 0.029498 | 78.271742 ± 1.165687 | 0.000678 ± 0.000004 | 0.352192 ± 0.035851 | 0.049978 ± 0.005197 | 0.321112 ± 0.037475 | 0.060406 ± 0.009619 | 0.439124 ± 0.048930 | 1.295618 ± 0.071699 | -0.784110 ± 0.095866 | 0.088008 ± 0.002805 |

## Branch-Usage Diagnostics

| Dataset | Method | Local top-K share | Global top-K share | Jaccard@20(Full, Local) | Jaccard@20(Full, Global) |
|---|---|---:|---:|---:|---:|
| Amazon-Book | SGL | 0.958240 ± 0.021518 | 0.041760 ± 0.021518 | 0.115700 ± 0.002267 | 0.060320 ± 0.015116 |
| Amazon-Book | XSimGCL | 0.923880 ± 0.037268 | 0.076120 ± 0.037268 | 0.116000 ± 0.004790 | 0.073100 ± 0.019106 |
| Yelp2018 | SGL | 0.999120 ± 0.000602 | 0.000880 ± 0.000602 | 0.201620 ± 0.006263 | 0.029540 ± 0.005503 |
| Yelp2018 | XSimGCL | 0.998000 ± 0.002101 | 0.002000 ± 0.002101 | 0.211880 ± 0.004924 | 0.032640 ± 0.007500 |
| Gowalla | SGL | 0.997040 ± 0.001623 | 0.002960 ± 0.001623 | 0.387460 ± 0.015840 | 0.015860 ± 0.002901 |
| Gowalla | XSimGCL | 0.996480 ± 0.002702 | 0.003520 ± 0.002702 | 0.377660 ± 0.006531 | 0.019020 ± 0.003031 |

## SGL Diagnostics

| Dataset | edge_drop_rate_view1 | edge_drop_rate_view2 | kept_edges_view1 | kept_edges_view2 | retained_edge_overlap_ratio | isolated_users_ratio_view1 | isolated_users_ratio_view2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Amazon-Book | 0.200008 ± 0.000011 | 0.199986 ± 0.000013 | 338037.840000 ± 32.189253 | 338047.260000 ± 35.563014 | 0.666672 ± 0.000008 | 0.000234 ± 0.000050 | 0.000194 ± 0.000046 |
| Yelp2018 | 0.050002 ± 0.000008 | 0.050000 ± 0.000000 | 1091652.660000 ± 166.301993 | 1091652.500000 ± 170.547339 | 0.904760 ± 0.000010 | 0.000006 ± 0.000013 | 0.000006 ± 0.000013 |
| Gowalla | 0.200014 ± 0.000017 | 0.200000 ± 0.000014 | 553740.220000 ± 266.522357 | 553749.500000 ± 257.672845 | 0.666664 ± 0.000022 | 0.005926 ± 0.000217 | 0.006010 ± 0.000595 |

## XSimGCL Diagnostics

| Dataset | noise_norm_mean | noise_norm_std | noise_to_signal_ratio_mean | clean_noisy_cos_mean | cl_layer | cl_layer_pos_sim_mean | cl_layer_neg_sim_mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Amazon-Book | 0.051354 ± 0.010881 | 0.022302 ± 0.006228 | 0.015352 ± 0.001530 | 0.999870 ± 0.000029 | 2.000000 ± 0.000000 | 0.594018 ± 0.034614 | 0.362412 ± 0.033894 |
| Yelp2018 | 0.049786 ± 0.019002 | 0.018230 ± 0.005605 | 0.017234 ± 0.000895 | 0.999858 ± 0.000022 | 2.000000 ± 0.000000 | 0.292278 ± 0.059314 | 0.272548 ± 0.063999 |
| Gowalla | 0.076072 ± 0.022965 | 0.026592 ± 0.003958 | 0.019036 ± 0.001208 | 0.999820 ± 0.000031 | 2.000000 ± 0.000000 | 0.352192 ± 0.035851 | 0.321112 ± 0.037475 |

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
- `loss_sup` is the supervised recommendation loss before adding the contrastive regularizer.
- `loss_cl` is the InfoNCE-style contrastive loss computed from the selected contrastive views.
- `loss_total` is the optimized objective, `loss_sup + lambda_cl * loss_cl`.
- `cl_to_sup_ratio` is `(lambda_cl * loss_cl) / loss_sup`, measuring the relative scale of the contrastive term.
- `user_cl_count_mean` and `item_cl_count_mean` are the average numbers of user and item anchors used in the contrastive objective per batch.
- `item_cl_coverage` is the fraction of deduplicated positive items in the supervised batch that are actually available as item-side contrastive anchors.
- `batches_with_empty_item_cl_ratio` is the fraction of batches in which item-side contrastive anchors are empty.
- `pos_sim_mean` and `pos_sim_std` summarize cosine similarities between matched positive contrastive pairs.
- `neg_sim_mean` and `neg_sim_std` summarize cosine similarities between unmatched contrastive pairs.
- `hardest_neg_sim_mean` is the mean maximum negative similarity for each anchor.
- `alignment_mean` is the mean squared distance between matched normalized contrastive views.
- `uniformity_mean` is the Wang-Isola uniformity score averaged across the two contrastive views; more negative values indicate more spread-out embeddings.
- `user_emb_std_per_dim_mean` and `item_emb_std_per_dim_mean` are the mean per-dimension standard deviations of user/item contrastive embeddings.
- `edge_drop_rate_view1` and `edge_drop_rate_view2` are the realized edge-drop ratios in the two SGL graph views.
- `kept_edges_view1` and `kept_edges_view2` are the numbers of message-passing edges retained in the two SGL views.
- `retained_edge_overlap_ratio` is the Jaccard overlap between the retained-edge masks of the two SGL views.
- `isolated_users_ratio_view1` and `isolated_users_ratio_view2` are the fractions of seed users with no retained outgoing message-passing edge in each SGL view.
- `noise_norm_mean` and `noise_norm_std` summarize the norm of the XSimGCL perturbation injected at the selected contrastive layer.
- `noise_to_signal_ratio_mean` is the mean ratio between perturbation norm and clean representation norm at the selected contrastive layer.
- `clean_noisy_cos_mean` is the mean cosine similarity between clean and perturbed layer representations.
- `cl_layer` is the intermediate GNN layer used as the contrastive view in XSimGCL.
- `cl_layer_pos_sim_mean` and `cl_layer_neg_sim_mean` are the positive and negative cosine-similarity means computed between the final view and the selected intermediate contrastive layer.
