import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgbm
import mlflow
from mlflow.cli import ui as mlflow_ui
from tqdm import tqdm

from params import PARAMS
from preprocess import label_encoding, process_date
import plot_funcs as pf
from utils import print_divider, flatten_dict, timestamp


def devide_by_sum(x):
    return x / x.sum()


def log_plot(args, plot_func, fp):
    if not isinstance(args, (tuple)):
        args = (args,)

    plot_func(*args, fp)
    mlflow.log_artifact(fp)
    os.remove(fp)
    print(f'Logged {fp}')


def main():
    X_train = pd.read_pickle('input/train.pkl').drop('building_id', axis=1).sample(frac=0.1, random_state=42)
    y_train = np.log1p(X_train.pop('meter_reading'))
    X_test = pd.read_pickle('input/test.pkl').drop(['row_id', 'building_id'], axis=1)
    process_date(X_train)
    process_date(X_test)
    label_encoding(X_train, X_test)

    models = []
    metrics = {'train': [], 'valid': []}
    feature_importances_split = np.zeros(X_train.shape[1])
    feature_importances_gain = np.zeros(X_train.shape[1])
    fold = KFold(**PARAMS['fold'])

    mlflow.set_experiment('ashrae-energy-prediction')

    with mlflow.start_run() as run:
        mlflow.log_params(flatten_dict(PARAMS))

        for fold_idx, (idx_tr, idx_val) in enumerate(fold.split(X_train)):
            print_divider(f'Fold {fold_idx}')
            X_tr, X_val = X_train.iloc[idx_tr], X_train.iloc[idx_val]
            y_tr, y_val = y_train.iloc[idx_tr], y_train.iloc[idx_val]

            model = lgbm.LGBMRegressor(**PARAMS['model'])
            model.fit(X_tr, y_tr,
                      eval_set=[(X_tr, y_tr), (X_val, y_val)],
                      eval_names=list(metrics.keys()),
                      **PARAMS['fit'])

            feature_importances_split += devide_by_sum(model.booster_.feature_importance(importance_type='split')) / fold.n_splits
            feature_importances_gain += devide_by_sum(model.booster_.feature_importance(importance_type='gain')) / fold.n_splits

            for key in metrics.keys():
                metrics[key].append({
                    'name': 'rmse',
                    'values': model.evals_result_[key]['rmse'],
                    'best_iteration': model.best_iteration_
                })

            models.append(model)

            del idx_tr, idx_val, X_tr, X_val, y_tr, y_val

        for key in metrics.keys():
            log_plot(metrics[key], pf.metric_history, f'metric_history_{key}.png')

        features = np.array(model.booster_.feature_name())
        log_plot((features, feature_importances_split, 'split'),
                 pf.feature_importance, 'feature_importance_split.png')
        log_plot((features, feature_importances_gain, 'gain'),
                 pf.feature_importance, 'feature_importance_gain.png')

    # prediction (split the data into chunks because it's too large)
    print_divider('prediction')
    i = 0
    pred = []
    chunk_size = 500000
    for j in tqdm(range(int(np.ceil(X_test.shape[0] / chunk_size)))):
        pred.append(np.expm1(sum([model.predict(X_test.iloc[i:i + chunk_size]) for model in models]) / fold.n_splits))
        i += chunk_size

    # make submission
    submission = pd.read_csv('input/sample_submission.csv')
    submission['meter_reading'] = pred
    submission.loc[submission['meter_reading'] < 0, 'meter_reading'] = 0
    submission_path = f'submission/{timestamp()}.csv'
    submission.to_csv(submission_path, index=False)
    mlflow.log_param('submission_path', submission_path)

    # run the MLflow UI to check the training result
    print_divider('MLflow UI')
    url_base = 'http://127.0.0.1:5000/#/experiments/{0}/runs/{1}'
    print('Run URL:', url_base.format(run.info.experiment_id, run.info.run_uuid))
    mlflow_ui()


if __name__ == '__main__':
    main()
