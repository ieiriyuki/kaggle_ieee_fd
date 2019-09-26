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

# +
import sys

from IPython.display import HTML
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# %matplotlib inline
pd.options.display.notebook_repr_html = True
# -

# %%time
columns = [
    "TransactionID", "isFraud", "TransactionDT", "card1", "card2", "card3",
    "card4", "card5", "card6", "addr1", "addr2"
]
train_transaction = pq.read_table("../data/train_transaction.gzip.pq",
                                  columns=columns).to_pandas().drop(
                                      [295458, 453416,
                                       570435]).set_index("TransactionID")
test_transaction = pq.read_table(
    "../data/test_transaction.gzip.pq",
    columns=columns).to_pandas().set_index("TransactionID")

print(train_transaction.shape, test_transaction.shape)

print(
    "train-card2: {}, card5: {}, addr1: {}, addr2: {}".format(
        len(train_transaction.card2.unique()),
        len(train_transaction.card5.unique()),
        len(train_transaction.addr1.unique()),
        len(train_transaction.addr2.unique())),
    "\ntest--card2: {}, card5: {}, addr1: {}, addr2: {}".format(
        len(test_transaction.card2.unique()),
        len(test_transaction.card5.unique()),
        len(test_transaction.addr1.unique()),
        len(test_transaction.addr2.unique())))

for i in ["card2", "card5"]:
    for j in ["addr1", "addr2"]:
        temp = train_transaction.loc[:, [i, j]].drop_duplicates()
        print(i, j, "train", len(train_transaction.loc[:, [i, j]].drop_duplicates()),
              "test", len(test_transaction.loc[:, [i, j]].drop_duplicates()))

df = train_transaction.loc[
    :, ["addr1", "addr2", "isFraud"]
].groupby(["addr1", "addr2"]).mean().sort_values("isFraud", ascending=False).join(
    train_transaction.loc[:, ["addr2", "addr1"]].assign(c=1).groupby(
        ["addr1", "addr2"]).count()
)
df.head(50)

df = pd.DataFrame().assign(
    addr2=train_transaction.addr2, c=1
).groupby("addr2").count().sort_values("c", ascending=False).join(
    train_transaction.loc[:, ["addr2", "isFraud"]].groupby("addr2").mean()
)
print(len(df))
df.head(20)

addr2s = df.index


def calc_hour(series):
    hours = series / 3600
    days = hours // 24
    return np.floor(hours) - days * 24


# +
fig, ax = plt.subplots(20, 2, figsize=(4, 32))
for i, addr in enumerate(addr2s):
    temp = train_transaction.loc[train_transaction.addr2 == addr, ["TransactionDT", "isFraud"]].copy()
    temp["hour"] = calc_hour(temp.TransactionDT)
    df = pd.DataFrame(index=[i for i in range(24)])
    df = df.join(
        temp.assign(c=1).loc[:, ["hour", "c"]].groupby("hour").count()).join(
        temp.loc[:, ["hour", "isFraud"]].groupby("hour").mean())
    
    if i == 20:
        break

    ax[i, 0].bar(df.index, df.c)
    ax[i, 0].set_xlim(0, 24)
    ax[i, 0].set_title(addr)
    ax[i, 1].plot(df.index.values, df.isFraud)
    ax[i, 1].set_xlim(2, 24)
    ax[i, 1].set_title(addr)
    
plt.tight_layout()
plt.show()


# +
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))
temp = train_transaction.loc[train_transaction.addr2 == 65, ["TransactionDT", "isFraud"]].copy()
temp["hour"] = calc_hour(temp.TransactionDT)
df = pd.DataFrame(index=[i for i in range(24)]).join(
    temp.assign(c=1).loc[:, ["hour", "c"]].groupby("hour").count()).join(
    temp.loc[:, ["hour", "isFraud"]].groupby("hour").mean())

ax1.bar(df.index, df.c)
ax1.set_xlim(0, 24)
ax1.set_title(65)
ax2.plot(df.index.values, df.isFraud)
ax2.set_xlim(2, 24)
ax2.set_title(65)
