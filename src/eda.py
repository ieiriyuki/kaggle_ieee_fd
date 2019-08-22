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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %matplotlib inline
# -

# %%time
train_transaction = pd.read_csv('../data/train_transaction.csv.zip',
                                dtype={'isFraud': bool,
                                       'TransactionDT': np.int32,
                                       'card1': np.int16}
                                index_col='TransactionID')
train_identity = pd.read_csv('../data/train_identity.csv.zip',
                             index_col='TransactionID')
test_transaction = pd.read_csv('../data/test_transaction.csv.zip',
                               dtype={'TransactionDT': np.int32,
                                      'card1': np.int16}
                               index_col='TransactionID')
test_identity = pd.read_csv('../data/test_identity.csv.zip',
                            index_col='TransactionID')

"""
train = train_transaction.join(train_identity)
test = test_transaction.join(test_identity)
"""

train_transaction.insert(
    train_transaction.shape[1], 'hour',
        (np.floor(train_transaction.TransactionDT / 3600) % 24).astype(np.int8))

vals = plt.hist(train_transaction.TransactionDT / (3600*24), bins=1800)
plt.xlim(70, 78)
plt.xlabel('Days')
plt.ylabel('Number of transactions')
plt.ylim(0,1000)
