"""
dataprep.py
Program responsible to model the data before training

authors: Kunal Nayyar & Srujan Shetty
"""

import pandas as pd
import numpy as np
import keras as K
from sklearn import preprocessing

# Globals
min_max_scaler = preprocessing.MinMaxScaler()


def get_train_data():
    # read training data - It is the aircraft engine run-to-failure data.
    train_df = pd.read_table("./CMAPSS_Data/train_FD003.txt", header=None, delim_whitespace=True)
    train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                        's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                        's15', 's16', 's17', 's18', 's19', 's20', 's21']

    train_df = train_df.sort_values(['id', 'cycle'])

    # Data Labeling - generate column RUL(Remaining Useful Life or Time to Failure)
    rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    train_df = train_df.merge(rul, on=['id'], how='left')
    train_df['RUL'] = train_df['max'] - train_df['cycle']
    train_df.drop('max', axis=1, inplace=True)
    train_df['cycle_norm'] = train_df['cycle']

    # MinMax normalization (from 0 to 1)
    cols_normalize = train_df.columns.difference(['id', 'cycle', 'RUL', 'label1', 'label2'])
    norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_df[cols_normalize]),
                                 columns=cols_normalize,
                                 index=train_df.index)
    temp_df = train_df[train_df.columns.difference(cols_normalize)].join(norm_train_df)
    train_df = temp_df.reindex(columns=train_df.columns)
    w1 = 30
    w0 = 15
    train_df['label1'] = np.where(train_df['RUL'] <= w1, 1, 0)
    train_df['label2'] = train_df['label1']
    train_df.loc[train_df['RUL'] <= w0, 'label2'] = 2

    return train_df

def get_test_data():
    # can be changed to another dataset
    test_df = pd.read_table("./CMAPSS_Data/test_FD003.txt", header=None, delim_whitespace=True)
    test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                       's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                       's15', 's16', 's17', 's18', 's19', 's20', 's21']

    # read ground truth
    truth_df = pd.read_csv('./CMAPSS_Data/RUL_FD003.txt', sep=" ", header=None)
    truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)
    cols_normalize = get_train_data().columns.difference({'id', 'cycle', 'RUL', 'label1', 'label2'})
    test_df['cycle_norm'] = test_df['cycle']

    # MinMax normalization (from 0 to 1)
    norm_test_df = pd.DataFrame(min_max_scaler.transform(test_df[cols_normalize]),
                                columns=cols_normalize,
                                index=test_df.index)
    temp_df = test_df[test_df.columns.difference(cols_normalize)].join(norm_test_df)
    test_df = temp_df.reindex(columns=test_df.columns)
    test_df = test_df.reset_index(drop=True)

    # We use the ground truth dataset to generate labels for the test data.
    rul = pd.DataFrame(test_df.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    truth_df.columns = ['more']
    truth_df['id'] = truth_df.index + 1
    truth_df['max'] = rul['max'] + truth_df['more']
    truth_df.drop('more', axis=1, inplace=True)

    test_df = test_df.merge(truth_df, on=['id'], how='left')
    test_df['RUL'] = test_df['max'] - test_df['cycle']
    test_df.drop('max', axis=1, inplace=True)

    return test_df


