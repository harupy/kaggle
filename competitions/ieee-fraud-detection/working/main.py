import re
import pandas as pd
import multiprocessing
import gc


def read_csv(file):
  return pd.read_csv(file)


def load_data():
  files = ['../input/train_identity.csv',
           '../input/train_transaction.csv',
           '../input/test_identity.csv',
           '../input/test_transaction.csv',
           '../input/sample_submission.csv']

  with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
    return pool.map(read_csv, files)


def select_columns_regexp(df, regexp):
  return [c for c in df.column if re.search(regexp, c)]


def select_cols_to_drop(df):
  one_value_cols = [col for col in df.columns if df[col].nunique() <= 1]
  many_null_cols = [col for col in df.columns if df[col].isnull().sum() / df.shape[0] > 0.9]
  big_top_value_cols = [col for col in df.columns if df[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
  return one_value_cols + many_null_cols + big_top_value_cols


def main():
  train_id, train_tr, test_id, test_tr, sub = load_data()

  train = pd.merge(train_tr, train_id, on='TransactionID', how='left')
  test = pd.merge(test_tr, test_id, on='TransactionID', how='left')

  del test_id, test_tr, train_id, train_tr
  gc.collect()

  cols_to_drop = list(set(select_cols_to_drop(train) + select_cols_to_drop(test)))
  cols_to_drop.remove('isFraud')
  print(cols_to_drop)

  train = train.drop(cols_to_drop, axis=1)
  test = test.drop(cols_to_drop, axis=1)


if __name__ == '__main__':
  main()
