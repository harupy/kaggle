import os
import gc
from collections import defaultdict, MutableMapping
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             confusion_matrix,
                             roc_curve,
                             roc_auc_score,
                             precision_recall_curve,
                             average_precision_score)
import lightgbm as lgbm
import mlflow
from mlflow.cli import ui as mlflow_ui
from config import (DATA_DIR,
                    SUBMISSION_DIR,
                    EXPRIMENT_PATH,
                    TRAIN_PATH,
                    TEST_PATH,
                    PARAMS)
from utils import make_timestamp, print_divider, flatten_dict
import plot_funcs as pf

warnings.filterwarnings('ignore', category=DeprecationWarning)


def make_submission(fp, proba):
    sub = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    sub['isFraud'] = proba
    sub.to_csv(fp, index=False)


def devide_by_sum(x):
    return x / x.sum()


def get_scores(y_true, y_pred):
    return {
      'accuracy': accuracy_score(y_true, y_pred),
      'precision': precision_score(y_true, y_pred),
      'recall': recall_score(y_true, y_pred),
      'f1': f1_score(y_true, y_pred),
    }


def log_plot(args, plot_func, fp):
    if not isinstance(args, (tuple)):
        args = (args,)

    plot_func(*args, fp)
    mlflow.log_artifact(fp)
    os.remove(fp)
    print(f'Logged {fp}')


def train_and_predict(X_train, y_train, X_test, params, experiment_path):
    # explicitly setting up mlflow experiment
    try:
        mlflow.create_experiment(experiment_path)
    except (mlflow.exceptions.RestException, mlflow.exceptions.MlflowException):
        print(f'The specified experiment ({experiment_path}) already exists.')

    skf = StratifiedKFold(**params['fold'])
    proba_oof = np.zeros(X_train.shape[0])
    pred_oof = np.zeros(X_train.shape[0])
    proba_test = np.zeros(X_test.shape[0])

    metric_history = []
    avg_scores = defaultdict(int)
    feature_importances_split = np.zeros(X_train.shape[1])
    feature_importances_gain = np.zeros(X_train.shape[1])

    with mlflow.start_run() as run:
        for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            print_divider(f'Fold: {fold}')
            X_trn, y_trn = X_train.iloc[trn_idx, :], y_train.iloc[trn_idx]
            X_val, y_val = X_train.iloc[val_idx, :], y_train.iloc[val_idx]
            model = lgbm.LGBMClassifier(**params['model'])
            model.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], eval_names=['valid'], **params['fit'])

            metric_history.append({
                'name': model.metric,
                'values': model.evals_result_['valid'][model.metric],
                'best_iteration': model.best_iteration_
            })

            feature_importances_split += devide_by_sum(model.booster_.feature_importance(importance_type='split')) / skf.n_splits
            feature_importances_gain += devide_by_sum(model.booster_.feature_importance(importance_type='gain')) / skf.n_splits

            proba_val = model.predict_proba(X_val, num_iteration=model.best_iteration_)[:, 1]
            pred_val = (proba_val > 0.5).astype(np.int8)
            proba_oof[val_idx] = proba_val
            pred_oof[val_idx] = pred_val
            scores_val = get_scores(y_val, pred_val)

            mlflow.log_metrics({
                **scores_val,
                'best_iteration': model.best_iteration_,
            }, step=fold)

            for k, v in scores_val.items():
                avg_scores[k] += v / skf.n_splits

            proba_test += model.predict_proba(X_test)[:, 1] / skf.n_splits

            del trn_idx, X_trn, y_trn, val_idx, X_val, y_val, proba_val, pred_val
            gc.collect()

        mlflow.log_params({
            **flatten_dict(params),
            'fold.strategy': skf.__class__.__name__,
            'model.type': model.__class__.__name__
        })

        print_divider('Saving plots')

        # scores
        log_plot(avg_scores, pf.scores, 'scores.png')

        # feature importance (only top 30)
        features = np.array(model.booster_.feature_name())
        log_plot((features, feature_importances_split, 'split', 30),
                 pf.feature_importance, 'feature_importance_split.png')
        log_plot((features, feature_importances_gain, 'gain', 30),
                 pf.feature_importance, 'feature_importance_gain.png')

        # metric history
        log_plot(metric_history, pf.metric_history, 'metric_history.png')

        # confusion matrix
        cm = confusion_matrix(y_train, pred_oof)
        log_plot(cm, pf.confusion_matrix, 'confusion_matrix.png')

        # roc curve
        fpr, tpr, _ = roc_curve(y_train, pred_oof)
        roc_auc = roc_auc_score(y_train, pred_oof)
        log_plot((fpr, tpr, roc_auc), pf.roc_curve, 'roc_curve.png')

        # precision-recall curve
        pre, rec, _ = precision_recall_curve(y_train, pred_oof)
        pr_auc = average_precision_score(y_train, pred_oof)
        log_plot((pre, rec, pr_auc), pf.pr_curve, 'pr_curve.png')

        return proba_test, run.info.experiment_id, run.info.run_uuid


def main():
    train = pd.read_pickle(TRAIN_PATH)
    X_test = pd.read_pickle(TEST_PATH)

    X_train = train.drop('isFraud', axis=1)
    y_train = train['isFraud']

    proba, experiment_id, run_uuid = train_and_predict(X_train, y_train, X_test, PARAMS, EXPRIMENT_PATH)
    timestamp = make_timestamp()
    make_submission(os.path.join(SUBMISSION_DIR, f'{timestamp}_{run_uuid}.csv'), proba)

    print_divider('MLflow UI')
    print('Run URL: http://127.0.0.1:5000/#/experiments/{0}/runs/{1}\n'
          .format(experiment_id, run_uuid))
    mlflow_ui()


if __name__ == '__main__':
    main()
