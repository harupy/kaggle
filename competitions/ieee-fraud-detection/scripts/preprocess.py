import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from config import DATA_DIR, TRAIN_PATH, TEST_PATH, PARAMS


def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def load_data(data_type):
    indentity = pd.read_csv(os.path.join(DATA_DIR, f'{data_type}_identity.csv'))
    transaction = pd.read_csv(os.path.join(DATA_DIR, f'{data_type}_transaction.csv'))
    return pd.merge(transaction, indentity, on='TransactionID', how='left')


def update_dtypes(df, dtypes):
    start_mem = df.memory_usage().sum() / 1024**2
    for col, dtype in tqdm(dtypes.items()):
        df[col] = df[col].astype(dtype)
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage: {:5.2f} Mb -> {:5.2f} Mb ({:.1f} % reduction)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))


def reduce_mem_usage(train, test, verbose=True):
    merged = pd.concat([train, test])
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    dtypes = {}
    for col in merged.columns:
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

    print('Updating train dtypes...')
    update_dtypes(train, dtypes)
    print('Updating test dtypes...')
    update_dtypes(test, dtypes)


def find_cols_to_drop(df):
    one_value = [col for col in df.columns if df[col].nunique() <= 1]
    many_null = [col for col in df.columns if df[col].isnull().sum() / df.shape[0] > 0.9]
    big_top_value = [col for col in df.columns if df[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
    return list(set(one_value + many_null + big_top_value))


def process_email(df):
    emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other', 'optonline.net': 'other',
              'hotmail.co.uk': 'microsoft', 'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo',
              'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 'aim.com': 'aol', 'hotmail.de': 'microsoft',
              'centurylink.net': 'centurylink', 'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other',
              'gmx.de': 'other', 'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 'protonmail.com': 'other',
              'hotmail.fr': 'microsoft', 'windstream.net': 'other', 'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo',
              'yahoo.de': 'yahoo', 'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other',
              'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft', 'verizon.net': 'yahoo',
              'msn.com': 'microsoft', 'q.com': 'centurylink', 'prodigy.net.mx': 'att', 'frontier.com': 'yahoo',
              'anonymous.com': 'other', 'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo',
              'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other', 'bellsouth.net': 'other',
              'embarqmail.com': 'centurylink', 'cableone.net': 'other', 'hotmail.es': 'microsoft', 'mac.com': 'apple',
              'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other',
              'cox.net': 'other', 'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}

    us_emails = ['gmail', 'net', 'edu']

    for c in ['P_emaildomain', 'R_emaildomain']:
        df[c + '_bin'] = df[c].map(emails)
        df[c + '_suffix'] = df[c].map(lambda x: str(x).split('.')[-1])
        df[c + '_suffix'] = df[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')


def process_domain(df):
    p = 'P_emaildomain'
    r = 'R_emaildomain'
    uknown = 'email_not_provided'
    df[p] = df[p].fillna(uknown)
    df[r] = df[r].fillna(uknown)
    df['email_check'] = np.where((df[p] == df[r]) & (df[p] != uknown), 1, 0)
    df[p + '_prefix'] = df[p].map(lambda x: x.split('.')[0])
    df[r + '_prefix'] = df[r].map(lambda x: x.split('.')[0])


def process_time(df):
    START_DATE = datetime.strptime('2017-11-30', '%Y-%m-%d')
    df['TransactionDT'] = df['TransactionDT'].fillna(df['TransactionDT'].median())
    df['dt'] = df['TransactionDT'].map(lambda x: (START_DATE + timedelta(seconds=x)))
    df['month'] = (df['dt'].dt.year - 2017) * 12 + df['dt'].dt.month
    df['woy'] = (df['dt'].dt.year - 2017) * 52 + df['dt'].dt.weekofyear
    df['dow'] = (df['dt'].dt.year - 2017) * 365 + df['dt'].dt.dayofyear

    df['hour'] = df['dt'].dt.hour
    df['dow'] = df['dt'].dt.dayofweek
    df['day'] = df['dt'].dt.day
    df.drop('dt', axis=1, inplace=True)


def process_D(df):
    df['D8'] = df['D8'].fillna(-1).astype(int)
    df['D8_not_same_day'] = np.where(df['D8'] >= 1, 1, 0)
    df['D9_not_na'] = np.where(df['D9'].isna(), 0, 1)
    df['D8_D9_decimal_dist'] = df['D8'].fillna(0) - df['D8'].fillna(0).astype(int)
    df['D8_D9_decimal_dist'] = ((df['D8_D9_decimal_dist'] - df['D9'])**2)**0.5


def impute_float(train, test):
    floats = ['float16', 'float32', 'float64']
    for col in train.columns:
        if train[col].dtype in floats or train[col].dtype in floats:
            merged = pd.concat([train[col], test[col]])
            train[col].fillna(merged.mean(), inplace=True)
            test[col].fillna(merged.mean(), inplace=True)


def label_encoding(train, test):
    train.fillna(-999, inplace=True)
    test.fillna(-999, inplace=True)

    for col in train.columns:
        if train[col].dtype == 'object' or test[col].dtype == 'object':
            le = LabelEncoder()
            le.fit(np.hstack((train[col].astype(str).values, test[col].astype(str).values)))
            train[col] = le.transform(train[col].astype(str).values)
            test[col] = le.transform(test[col].astype(str).values)


def freq_encoding(train, test):
    columns = ['card1', 'card2', 'card3', 'card5',
               'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14',
               'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8',
               'addr1', 'addr2',
               'dist1', 'dist2',
               'P_emaildomain', 'R_emaildomain',
               'DeviceInfo', 'device_name',
               'id_30', 'id_33',
               'uid', 'uid2', 'uid3',
               ]

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


def assert_columns_equal(train, test):
    ncols_tr, ncols_te = len(train.columns), len(test.columns)
    assert ncols_tr == ncols_te, 'columns length must be equal: (train: {}, test: {}).'.format(ncols_tr, ncols_te)

    diff_cols = [(col_tr, col_te) for col_tr, col_te in zip(train.columns, test.columns) if col_tr != col_te]
    assert len(diff_cols) == 0, 'columns order must be equal: {}.'.format(diff_cols)


def assert_dtypes_equal(train, test):
    diff_dtypes = [(col, dt_tr, dt_te) for col, dt_tr, dt_te in zip(train.columns, train.dtypes, test.dtypes) if dt_tr != dt_te]
    assert len(diff_dtypes) == 0, 'dtypes must be equal: {}'.format(diff_dtypes)


def preprocess():
    train = load_data('train')
    y_train = train.pop('isFraud')
    test = load_data('test')
    assert_columns_equal(train, test)

    reduce_mem_usage(train, test)
    assert_dtypes_equal(train, test)

    cols_to_drop = list(set(find_cols_to_drop(train) + find_cols_to_drop(test)))
    train.drop(cols_to_drop, axis=1, inplace=True)
    test.drop(cols_to_drop, axis=1, inplace=True)
    print(f'Dropped {len(cols_to_drop)} columns.')

    # functions which can be applied separately
    funcs_single = [
        process_domain,
        process_email,
        process_time,
        process_D,
    ]

    for func in funcs_single:
        print('Applying:', func.__name__)
        func(train)
        func(test)

    cols = ['TransactionID', 'TransactionDT']
    train.drop(cols, axis=1, inplace=True)
    test.drop(cols, axis=1, inplace=True)

    # functions which can't be applied separately
    funcs_both = [
        impute_float,
        label_encoding,
        freq_encoding,
    ]

    for func in funcs_both:
        print('Applying:', func.__name__)
        func(train, test)

    target_encoding(train, y_train, test)

    reduce_mem_usage(train, test)
    train['isFraud'] = y_train.astype(np.int8)

    train.to_pickle(TRAIN_PATH)
    test.to_pickle(TEST_PATH)


if __name__ == '__main__':
    preprocess()
