import os
import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def update_dtypes(df, dtypes):
    start_mem = df.memory_usage().sum() / 1024**2
    for col, dtype in dtypes.items():
        if col not in df.columns:
            continue
        df[col] = df[col].astype(dtype)
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage: {:5.2f} Mb -> {:5.2f} Mb ({:.1f} % reduction)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))


def reduce_mem_usage(*dfs, verbose=True):
    merged = pd.concat(dfs, axis=0)
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    dtypes = {}
    for col in tqdm(merged.columns):
        col_type = merged[col].dtypes
        if col_type in numerics:
            c_min = merged[col].min()
            c_max = merged[col].max()
            if str(col_type).startswith('int'):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    dtypes[col] = np.int8
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    dtypes[col] = np.int16
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    dtypes[col] = np.int32
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    dtypes[col] = np.int64
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    dtypes[col] = np.float16
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    dtypes[col] = np.float32
                else:
                    dtypes[col] = np.float64

    for idx, df in enumerate(dfs):
        print(f'Updating dataframe {idx}...')
        update_dtypes(df, dtypes)


def process_date(df):
    ts = df.pop('timestamp')
    df['year'] = ts.dt.year.astype(np.uint16)
    df['month'] = ts.dt.month.astype(np.uint16)
    df['day'] = ts.dt.day.astype(np.uint8)
    df['weekofyear'] = ts.dt.weekofyear.astype(np.uint8)
    df['dayofweek'] = ts.dt.dayofweek.astype(np.uint8)


def find_cols_to_drop(df):
    one_value = [col for col in df.columns if df[col].nunique() <= 1]
    many_null = [col for col in df.columns if df[col].isnull().sum() / df.shape[0] > 0.9]
    big_top_value = [col for col in df.columns if df[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
    return list(set(one_value + many_null + big_top_value))


def impute_float(train, test):
    floats = ['float16', 'float32', 'float64']
    for col in train.columns:
        if train[col].dtype in floats or train[col].dtype in floats:
            merged = pd.concat([train[col], test[col]])
            train[col].fillna(merged.mean(), inplace=True)
            test[col].fillna(merged.mean(), inplace=True)


def label_encoding(train, test):
    # train.fillna(-999, inplace=True)
    # test.fillna(-999, inplace=True)

    for col in train.columns:
        if train[col].dtype == 'object' or test[col].dtype == 'object':
            le = LabelEncoder()
            le.fit(np.hstack((train[col].astype(str).values, test[col].astype(str).values)))
            train[col] = le.transform(train[col].astype(str).values)
            test[col] = le.transform(test[col].astype(str).values)


def freq_encoding(train, test):
    for col in columns:
        if col in train.columns and col in test.columns:
            merged = pd.concat([train[col], test[col]])
            vc = merged.value_counts(dropna=False)
            train[col + '_freq_enc'] = train[col].map(vc)
            test[col + '_freq_enc'] = test[col].map(vc)


def target_encoding(X_train, y_train, X_test):
    columns = ['D1', 'D2', 'D3']
    skf = StratifiedKFold(**PARAMS['fold'])
    for idx_tr, idx_val in skf.split(X_train, y_train):
        X_tr, y_tr = X_train.iloc[idx_tr, :], y_train.iloc[idx_tr]
        X_val, _ = X_train.iloc[idx_val, :], y_train.iloc[idx_val]

        for col in columns:
            mapper = X_tr.assign(target=y_tr).groupby(col)['target'].mean()
            X_train.loc[idx_val, f'{col}_target_enc'] = X_val[col].map(mapper)


def preprocess():
    pass


if __name__ == '__main__':
    preprocess()
