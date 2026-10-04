import pandas as pd
from types import SimpleNamespace
import typing as t
from scipy import sparse as sp
import numpy as np
import random
from ast import literal_eval as make_tuple

np.random.seed(42)
random.seed(42)

"""
prefiltering:
    strategy: global_threshold|user_average|user_k_core|item_k_core|iterative_k_core|n_rounds_k_core|cold_users
    threshold: 3|average
    core: 5
    rounds: 2
"""

class NegativeSampler:

    @staticmethod
    def sample(ns: SimpleNamespace, public_users: t.Dict, public_items: t.Dict, private_users: t.Dict,
               private_items: t.Dict, i_train: sp.csr_matrix,
               val: t.Dict = None, test: t.Dict = None):

        val_negative_items, val_negative_items_set = NegativeSampler.process_sampling(
            ns, public_users, public_items, private_users,
            private_items, i_train,
            val, validation=True
        ) if val is not None else (None, None)

        test_negative_items, test_negative_items_set = NegativeSampler.process_sampling(ns, public_users, public_items, private_users,
                                                              private_items, i_train,
                                                               test) if test != None else (None, None)

        if val_negative_items_set and test_negative_items_set:
            selected_items = val_negative_items_set.union(test_negative_items_set)

        elif val_negative_items_set:
            selected_items = val_negative_items_set

        elif test_negative_items_set:
            selected_items = test_negative_items_set

        else:
            selected_items = None

        return (val_negative_items, test_negative_items, selected_items) if val_negative_items else (test_negative_items, test_negative_items, selected_items)

    @staticmethod
    def process_sampling(ns: SimpleNamespace, public_users: t.Dict, public_items: t.Dict, private_users: t.Dict,
                         private_items: t.Dict, i_train: sp.csr_matrix,
                         test: t.Dict, validation=False) -> sp.csr_matrix:
        # quando è fixed questo non viene usato
        # candidate_negatives = ((i_test + i_train).astype('bool') != True)
        ns = ns.negative_sampling

        strategy = getattr(ns, "strategy", None)

        if strategy == "random":
            raise NotImplementedError(
                "Negative sampling strategy 'random' is not supported in this patched pipeline. "
                "Use strategy 'fixed'."
            )
        elif strategy == "fixed":
            files = getattr(ns, "files", None)
            if files is not None:
                if not isinstance(files, list):
                    files = [files]
                file_ = files[0] if validation == False else files[1]
                negative_items = NegativeSampler.read_from_files(public_users, public_items, file_)
            pass
        else:
            raise Exception("Missing strategy")

        return negative_items

    @staticmethod
    def read_from_files(public_users: t.Dict, public_items: t.Dict, filepath: str):
        def parse_row(row):
            row = [x for x in row.tolist() if pd.notna(x)]
            user_id, pos_item_id = make_tuple(row[0])
            negatives = [int(x) for x in row[1:]]
            candidate_items = [int(pos_item_id)] + negatives
            
            return int(user_id), candidate_items

        df = pd.read_csv(filepath, sep='\t', header=None)
        parsed = df.apply(lambda x: pd.Series(parse_row(x)), axis=1)
        parsed.columns = ['user', 'candidate_items']

        flat_items = [it for items in parsed['candidate_items'] for it in items]
        selected_items = set(flat_items)

        return parsed.set_index('user').to_dict()['candidate_items'], selected_items

    @staticmethod
    def build_sparse(map_ : t.Dict, nusers: int, nitems: int):

        rows_cols = [(u, i) for u, items in map_.items() for i in items.keys()]
        rows = [u for u, _ in rows_cols]
        cols = [i for _, i in rows_cols]
        data = sp.csr_matrix((np.ones_like(rows), (rows, cols)), dtype='float32',
                             shape=(nusers, nitems))
        return data