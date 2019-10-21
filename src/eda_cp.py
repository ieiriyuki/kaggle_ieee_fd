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
import re

from IPython.display import HTML
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %matplotlib inline
pd.options.display.notebook_repr_html = True
# -

# %%time
train_transaction = pd.read_csv('../data/train_transaction.csv.zip',
                                dtype={'isFraud': bool,
                                       'TransactionDT': np.int32,
                                       'card1': np.int16},
                                index_col='TransactionID',
                                skiprows=lambda x: x in [295458, 453416, 570435],
                                usecols=lambda x: x not in ['D9'])

train_transaction.columns

train_transaction["isTest"] = 0

test_transaction = pd.read_csv('../data/test_transaction.csv.zip',
                               dtype={'TransactionDT': np.int32,
                                      'card1': np.int16},
                               index_col='TransactionID',
                               usecols=lambda x: x not in ['D9'])

test_transaction["isFraud"] = None
test_transaction["isTest"] = 1

train_identity = pd.read_csv('../data/train_identity.csv.zip',
                             index_col='TransactionID')

test_identity = pd.read_csv('../data/test_identity.csv.zip',
                            index_col='TransactionID',
                            usecols=lambda x: x not in ['D9'])

# train_transaction.shape = (590540, 392)

train = train_transaction.join(train_identity)

test = test_transaction.join(test_identity)

vals = plt.hist(train_transaction.TransactionDT / (3600*24), bins=1800)
plt.xlim(70, 78)
plt.xlabel('Days')
plt.ylabel('Number of transactions')
plt.ylim(0,1000)

train_transaction.P_emaildomain.nunique()

subst(a)


def subst(x):
    return re.sub(r'.+\.', '', x)


aaa = pd.DataFrame(
        {"last": train_transaction.P_emaildomain.str.replace('.+\.', '')},
        index=train_transaction.P_emaildomain).drop_duplicates()

train_transaction.loc[:, ["P_emaildomain", "isFraud"]].groupby("P_emaildomain").mean().join(
    train_transaction.loc[:, ["P_emaildomain", "card1"]].groupby("P_emaildomain").count(),
    how="left").join(pd.DataFrame(
        {"last": train_transaction.P_emaildomain.str.replace('.+\.', '')},
        index=train_transaction.P_emaildomain).drop_duplicates(), how="left").sort_values("card1", ascending=False)

pem = {i for i in train_transaction.P_emaildomain.dropna().unique()}
pem
