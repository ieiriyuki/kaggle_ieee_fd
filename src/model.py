#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier as rfc
import xgboost as xgb


"""
>>> from sklearn import metrics
>>> scores = cross_val_score(
...     clf, iris.data, iris.target, cv=5, scoring='f1_macro')
>>> scores
array([0.96..., 1.  ..., 0.96..., 0.96..., 1.        ])
"""


def main():

    train_transaction = pd.read_csv('./data/train_transaction.csv.zip',
                                    dtype={'isFraud': bool,
                                           'TransactionDT': np.int32,
                                           'card1': np.int16},
                                    index_col='TransactionID',
                                    skiprows=lambda x: x in [295458,
                                                             453416,
                                                             570435],
                                    usecols=lambda x: x not in ['D9'])
    train_identity = pd.read_csv('./data/train_identity.csv.zip',
                                 index_col='TransactionID')
    test_transaction = pd.read_csv('./data/test_transaction.csv.zip',
                                   dtype={'TransactionDT': np.int32,
                                          'card1': np.int16},
                                   index_col='TransactionID',
                                   usecols=lambda x: x not in ['D9'])
    test_identity = pd.read_csv('./data/test_identity.csv.zip',
                                index_col='TransactionID',
                                usecols=lambda x: x not in ['D9'])
    sample_submission = pd.read_csv('./data/sample_submission.csv.zip',
                                    index_col='TransactionID')

    train = train_transaction.merge(
        train_identity, how='left', left_index=True, right_index=True)
    test = test_transaction.merge(
        test_identity, how='left', left_index=True, right_index=True)

    y_train = train['isFraud'].copy()

    # Drop target, fill in NaNs
    X_train = train.drop('isFraud', axis=1).fillna(-999)
    X_test = test.copy().fillna(-999)

    del train, test, train_transaction, \
        train_identity, test_transaction, test_identity

    # Label Encoding
    for f in X_train.columns:
        if X_train[f].dtype=='object' or X_test[f].dtype=='object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(X_train[f].values) + list(X_test[f].values))
            X_train[f] = lbl.transform(list(X_train[f].values))
            X_test[f] = lbl.transform(list(X_test[f].values))

    clf = xgboost(X_train, y_train)
    sample_submission['isFraud'] = clf.predict_proba(X_test)[:, 1]

    return 1


def randomforest(X_train, y_train):
    param_rfc = {'n_estimators': 500,
                 'max_depth': 6,
                 'min_samples_split': 4,
                 'min_samples_leaf': 2,
                 'max_features': 0.5,
                 'n_jobs': 4}
    clf_rfc = rfc(**param_rfc)
    clf_rfc.fit(X_train, y_train)
    return clf_rfc


def xgboost(X_train, y_train):
    clf = xgb.XGBClassifier(n_estimators=500,
                            n_jobs=4,
                            max_depth=6,
                            learning_rate=0.05,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            missing=-999)
    clf.fit(X_train, y_train)
    return clf


if __name__ == "__main__":
    main()
