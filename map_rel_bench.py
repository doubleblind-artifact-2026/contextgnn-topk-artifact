"""
This file is adapted from code originally released in the ContextGNN project.

Original source:
- https://github.com/kumo-ai/ContextGNN

The original code is distributed under the MIT License. The corresponding
copyright and license notices are preserved in this repository.

Modifications were made for the experiments accompanying this submission.
"""

import numpy as np
import argparse
import os.path as osp
import pandas as pd
from torch_geometric.seed import seed_everything

PSEUDO_TIME = "pseudo_time"
TRAIN_SET_TIMESTAMP = pd.Timestamp("1970-01-01")
SRC_ENTITY_TABLE = "user_table"
DST_ENTITY_TABLE = "item_table"
TRANSACTION_TABLE = "transaction_table"
SRC_ENTITY_COL = "user_id"
DST_ENTITY_COL = "item_id"

SPLIT_SEED = 42
VAL_LEAVE_K = 1
VAL_SET_TIMESTAMP = TRAIN_SET_TIMESTAMP + pd.Timedelta(days=1)
TEST_SET_TIMESTAMP = TRAIN_SET_TIMESTAMP + pd.Timedelta(days=2)

def ts_to_unix_seconds(ts: pd.Timestamp) -> int:
    return int(ts.value // 10**9)

parser = argparse.ArgumentParser(description="Run sample main")
parser.add_argument("--dataset", type=str, default="amazon-book")
args = parser.parse_args()

dataset = args.dataset
input_data_dir = f"./data/{dataset}/"

seed_everything(SPLIT_SEED)

def split_train_into_train_val(df: pd.DataFrame, leave_k_out: int = 1):
    rng = np.random.default_rng(SPLIT_SEED)

    remaining_item_freq = (
        df.explode(DST_ENTITY_COL)[DST_ENTITY_COL]
        .value_counts()
        .to_dict()
    )

    train_rows = []
    val_rows = []

    for row in df.itertuples(index=False):
        user_id = getattr(row, SRC_ENTITY_COL)
        items = list(getattr(row, DST_ENTITY_COL))

        max_holdout = min(leave_k_out, max(len(items) - 1, 0))

        eligible_positions = [
            j for j, it in enumerate(items)
            if remaining_item_freq.get(it, 0) > 1
        ]

        if max_holdout > 0 and eligible_positions:
            chosen = rng.choice(
                eligible_positions,
                size=min(max_holdout, len(eligible_positions)),
                replace=False
            )

            chosen = set(np.atleast_1d(chosen).tolist())
        else:
            chosen = set()

        train_items = []
        val_items = []

        for j, it in enumerate(items):
            if j in chosen:
                val_items.append(it)
                remaining_item_freq[it] -= 1
            else:
                train_items.append(it)

        train_rows.append({
            SRC_ENTITY_COL: user_id,
            DST_ENTITY_COL: train_items
        })

        if val_items:
            val_rows.append({
                SRC_ENTITY_COL: user_id,
                DST_ENTITY_COL: val_items
            })

    return pd.DataFrame(train_rows), pd.DataFrame(val_rows)

######################
### LOAD USER DATA ###
######################

user_path = osp.join(input_data_dir, "user_list.txt")
src_df = pd.read_csv(user_path, delim_whitespace=True)
src_df = src_df.drop(columns=["org_id"]).rename(columns={"remap_id": SRC_ENTITY_COL}) # Drop org_id and rename remap_id to user_id
src_df[PSEUDO_TIME] = ts_to_unix_seconds(TRAIN_SET_TIMESTAMP)

######################
### LOAD ITEM DATA ###
######################

item_path = osp.join(input_data_dir, "item_list.txt")
dst_df = pd.read_csv(item_path, delim_whitespace=True)
dst_df = dst_df.drop(columns=["org_id"]).rename(columns={"remap_id": DST_ENTITY_COL}) # Drop org_id and rename remap_id to item_id
dst_df[PSEUDO_TIME] = ts_to_unix_seconds(TRAIN_SET_TIMESTAMP)

##########################################
### LOAD USER-ITEM LINK FOR TRAIN DATA ###
##########################################

train_path = osp.join(input_data_dir, "train.txt")

user_ids = []
item_ids = []

with open(train_path, "r") as file:
    for line in file:
        values = list(map(int, line.split()))

        user_id = values[0]
        item_ids_for_user = values[1:]

        user_ids.append(user_id)
        item_ids.append(item_ids_for_user)

train_full_df = pd.DataFrame({SRC_ENTITY_COL: user_ids, DST_ENTITY_COL: item_ids})
train_full_df = train_full_df.sample(frac=1, random_state=SPLIT_SEED).reset_index(drop=True)

train_df, val_df = split_train_into_train_val(train_full_df, leave_k_out=VAL_LEAVE_K)

train_df[PSEUDO_TIME] = ts_to_unix_seconds(TRAIN_SET_TIMESTAMP)
val_df[PSEUDO_TIME] = ts_to_unix_seconds(VAL_SET_TIMESTAMP)

#########################################
### LOAD USER-ITEM LINK FOR TEST DATA ###
#########################################

test_path = osp.join(input_data_dir, "test.txt")

user_ids = []
item_ids = []

with open(test_path, "r") as file:
    for line in file:
        values = list(map(int, line.split()))

        user_id = values[0]
        item_ids_for_user = values[1:]

        user_ids.append(user_id)
        item_ids.append(item_ids_for_user)

test_df = pd.DataFrame({"user_id": user_ids, "item_id": item_ids})
test_df = test_df.sample(frac=1, random_state=SPLIT_SEED).reset_index(drop=True)

test_df[PSEUDO_TIME] = ts_to_unix_seconds(TEST_SET_TIMESTAMP)

#########################
### EXPLOSION OF DATA ###
#########################

train_df_explode = train_df.explode(DST_ENTITY_COL).reset_index(drop=True)
target_df = train_df_explode

train_users_vocab = set(target_df[SRC_ENTITY_COL].unique())
train_items_vocab = set(target_df[DST_ENTITY_COL].unique())

val_df = val_df[val_df[SRC_ENTITY_COL].isin(train_users_vocab)].copy()
test_df = test_df[test_df[SRC_ENTITY_COL].isin(train_users_vocab)].copy()

val_df[DST_ENTITY_COL] = val_df[DST_ENTITY_COL].map(lambda xs: [it for it in xs if it in train_items_vocab])
test_df[DST_ENTITY_COL] = test_df[DST_ENTITY_COL].map(lambda xs: [it for it in xs if it in train_items_vocab])

val_df = val_df[val_df[DST_ENTITY_COL].map(len) > 0].reset_index(drop=True)
test_df = test_df[test_df[DST_ENTITY_COL].map(len) > 0].reset_index(drop=True)

val_df_explode = val_df.explode(DST_ENTITY_COL).reset_index(drop=True)
test_df_explode = test_df.explode(DST_ENTITY_COL).reset_index(drop=True)

train_items_vocab = set(target_df[DST_ENTITY_COL].unique())
val_cold_items = sorted(set(val_df_explode[DST_ENTITY_COL].unique()) - train_items_vocab)
test_cold_items = sorted(set(test_df_explode[DST_ENTITY_COL].unique()) - train_items_vocab)

print("Cold items in validation:", len(val_cold_items), val_cold_items[:20])
print("Cold items in test:", len(test_cold_items), test_cold_items[:20])

######################
### DISJOINT CHECK ###
######################

train_pairs = set(map(tuple, target_df[[SRC_ENTITY_COL, DST_ENTITY_COL]].values.tolist()))
val_pairs = set(map(tuple, val_df_explode[[SRC_ENTITY_COL, DST_ENTITY_COL]].values.tolist()))
test_pairs = set(map(tuple, test_df_explode[[SRC_ENTITY_COL, DST_ENTITY_COL]].values.tolist()))

assert train_pairs.isdisjoint(val_pairs), "Train/Validation overlap detected."
assert val_pairs.isdisjoint(test_pairs), "Validation/Test overlap detected."
assert train_pairs.isdisjoint(test_pairs), "Train/Test overlap detected."

#################################################################
### SAVE train_elliot.tsv AND ALL THE INTERMEDIATE DATAFRAMES ###
#################################################################

src_df = src_df[src_df[SRC_ENTITY_COL].isin(train_users_vocab)].reset_index(drop=True)
dst_df = dst_df[dst_df[DST_ENTITY_COL].isin(train_items_vocab)].reset_index(drop=True)

target_df = target_df[
    target_df[SRC_ENTITY_COL].isin(train_users_vocab) &
    target_df[DST_ENTITY_COL].isin(train_items_vocab)
].reset_index(drop=True)

target_df[["user_id", "item_id"]].to_csv(input_data_dir + "train_elliot.tsv", sep="\t", index=False, header=False)
val_df_explode[["user_id", "item_id"]].to_csv(input_data_dir + "val_elliot.tsv", sep="\t", index=False, header=False)

train_df = train_df[train_df[SRC_ENTITY_COL].isin(train_users_vocab)].copy()
train_df[DST_ENTITY_COL] = train_df[DST_ENTITY_COL].map(lambda xs: [it for it in xs if it in train_items_vocab])
train_df = train_df[train_df[DST_ENTITY_COL].map(len) > 0].reset_index(drop=True)

with open(input_data_dir + "test_elliot.tsv", "w") as file:
    for idx, row in test_df.iterrows():
        for it in row["item_id"]:
            file.write(str(row["user_id"])+"\t"+str(it)+"\n")

src_df.to_parquet(input_data_dir + "src_df.tsv", engine="pyarrow", index=False)
dst_df.to_parquet(input_data_dir + "dst_df.tsv", engine="pyarrow", index=False)
target_df.to_parquet(input_data_dir + "target_df.tsv", engine="pyarrow", index=False)
test_df.to_parquet(input_data_dir + "test_df.tsv", engine="pyarrow", index=False)
train_df.to_parquet(input_data_dir + "train_df.tsv", engine="pyarrow", index=False)
val_df.to_parquet(input_data_dir + "val_df.tsv", engine="pyarrow", index=False)