# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

# %%time
columns = [
    "TransactionID", "isFraud", "TransactionDT"
]
train_transaction = pq.read_table("../data/train_transaction.gzip.pq",
                                  columns=columns).to_pandas().drop(
                                      [295458, 453416,
                                       570435]).set_index("TransactionID")
test_transaction = pq.read_table(
    "../data/test_transaction.gzip.pq",
    columns=columns).to_pandas().set_index("TransactionID")

print(train_transaction.shape, test_transaction.shape)

l = len(train_transaction)
split_bound = int(l * 0.9)
X_train, X_test, y_train, y_test = (
    train_transaction.iloc[:split_bound, :].drop("isFraud", axis=1).copy(),
    train_transaction.iloc[split_bound:, :].drop("isFraud", axis=1).copy(),
    train_transaction.iloc[:split_bound, 0].copy(),
    train_transaction.iloc[split_bound:, 0].copy())

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


def time_series_spliter(data: pd.DataFrame, n_split=5, proportion=0.1):
    l = len(data)
    s = int(l * proportion)
    for i in range(n_split):
        size = l -(n_split - i) * s
        yield np.arange(size), np.arange(size, size + s)


for i in time_series_spliter(X_train):
    print("train pct: {:.05f}, valid pct: {:.05f}".format(
        y_train.iloc[i[0]].mean(), y_train.iloc[i[1]].mean()
    ))
