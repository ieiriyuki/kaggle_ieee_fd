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
from IPython.display import HTML
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# %matplotlib inline
pd.options.display.notebook_repr_html = True
# -

# %%time
columns = [
    "TransactionID", "isFraud", "P_emaildomain", "R_emaildomain"
]
train_transaction = pq.read_table("../data/train_transaction.gzip.pq",
                                  columns=columns).to_pandas().drop(
                                      [295458, 453416,
                                       570435]).set_index("TransactionID")
test_transaction = pq.read_table(
    "../data/test_transaction.gzip.pq",
    columns=columns).to_pandas().set_index("TransactionID")

train_transaction = train_transaction.fillna("NA")
test_transaction = test_transaction.fillna("NA")

domains = np.union1d(train_transaction.P_emaildomain.unique(),
                     train_transaction.R_emaildomain.unique())
domains = np.union1d(domains, test_transaction.P_emaildomain.unique())
domains = np.union1d(domains, test_transaction.R_emaildomain.unique())

domains_stats = pd.DataFrame({"domains": domains})

domains_stats.head()

domains_stats = domains_stats.join(
    train_transaction.loc[:, ["isFraud", "P_emaildomain"]].groupby("P_emaildomain").count().rename(
        columns={"isFraud": "trainP_cnt"}), on="domains").join(
    train_transaction.loc[:, ["isFraud", "P_emaildomain"]].groupby("P_emaildomain").mean().rename(
        columns={"isFraud": "trainP_pct"}), on="domains").join(
    train_transaction.loc[:, ["isFraud", "R_emaildomain"]].groupby("R_emaildomain").count().rename(
        columns={"isFraud": "trainR_cnt"}), on="domains").join(
    train_transaction.loc[:, ["isFraud", "R_emaildomain"]].groupby("R_emaildomain").mean().rename(
        columns={"isFraud": "trainR_pct"}), on="domains")

domains_stats.sort_values("trainP_cnt", ascending=False).head(20)

# +
temp = domains_stats.sort_values("trainP_cnt", ascending=False)
fig, ax = plt.subplots(figsize=(20, 8))
def format_fn(tick_val, tick_pos):
    if int(tick_val) in np.arange(len(temp)):
        return temp.domains.iloc[int(tick_val)]
    else:
        return ''

ax.xaxis.set_major_formatter(FuncFormatter(format_fn))
ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=999))
ax.tick_params(rotation=90, labelsize=18)
ax.bar(np.arange(len(temp)), temp.trainP_cnt)
ax2 = ax.twinx()
ax2.plot(np.arange(len(temp)), temp.trainP_pct, color="orange")
# -

domains_stats.sort_values("trainP_pct", ascending=False).head(20)

domains_stats.sort_values("trainR_cnt", ascending=False).head(20)

# +
temp = domains_stats.sort_values("trainR_cnt", ascending=False)
fig, ax = plt.subplots(figsize=(20, 8))
def format_fn(tick_val, tick_pos):
    if int(tick_val) in np.arange(len(temp)):
        return temp.domains.iloc[int(tick_val)]
    else:
        return ''

ax.xaxis.set_major_formatter(FuncFormatter(format_fn))
ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=999))
ax.tick_params(rotation=90, labelsize=18)
ax.bar(np.arange(len(temp)), temp.trainR_cnt)
ax2 = ax.twinx()
ax2.plot(np.arange(len(temp)), temp.trainR_pct, color="orange")
# -

domains_stats.sort_values("trainR_pct", ascending=False).head(20)

emails = {'aim.com': 'aol',
          'anonymous.com': 'other',
          'aol.com': 'aol',
          'att.net': 'att',
          'bellsouth.net': 'other',
          'cableone.net': 'other',
          'centurylink.net': 'centurylink',
          'cfl.rr.com': 'other',
          'charter.net': 'spectrum',
          'comcast.net': 'other',
          'cox.net': 'other',
          'earthlink.net': 'other',
          'embarqmail.com': 'centurylink',
          'frontier.com': 'yahoo',
          'frontiernet.net': 'yahoo',
          'gmail': 'google',
          'gmail.com': 'google',
          'gmx.de': 'other',
          'hotmail.com': 'microsoft',
          'hotmail.co.uk': 'microsoft',
          'hotmail.de': 'microsoft',
          'hotmail.es': 'microsoft',
          'hotmail.fr': 'microsoft',
          'icloud.com': 'apple'
          'juno.com': 'other',
          'live.com': 'microsoft',
          'live.com.mx': 'microsoft',
          'live.fr': 'microsoft',
          'mac.com': 'apple',
          'mail.com': 'mail',
          'me.com': 'apple',
          'msn.com': 'microsoft',
          'netzero.com': 'other',
          'netzero.net': 'other',
          'optonline.net': 'other',
          'outlook.com': 'outlook',
          'outlook.es': 'outlook',
          'prodigy.net.mx': 'att',
          'protonmail.com': 'proton',
          'ptd.net': 'other',
          'q.com': 'centurylink',
          'roadrunner.com': 'other',
          'rocketmail.com': 'yahoo',
          'sbcglobal.net': 'att',
          'sc.rr.com': 'other',
          'scranton.edu': 'other',
          'servicios-ta.com': 'other',
          'suddenlink.net': 'other',
          'twc.com': 'spectrum',
          'verizon.net': 'yahoo',
          'web.de': 'other',
          'windstream.net': 'other',
          'yahoo.co.jp': 'yahoo',
          'yahoo.co.uk': 'yahoo',
          'yahoo.com': 'yahoo',
          'yahoo.com.mx': 'yahoo',
          'yahoo.de': 'yahoo',
          'yahoo.es': 'yahoo',
          'yahoo.fr': 'yahoo',
          'ymail.com': 'yahoo'}
