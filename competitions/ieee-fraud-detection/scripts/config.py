import os
DATA_DIR = 'input'
SUBMISSION_DIR = 'submission'
EXPRIMENT_PATH = 'ieee-fraud-detection'
TRAIN_PATH = os.path.join(DATA_DIR, 'train.pkl')
TEST_PATH = os.path.join(DATA_DIR, 'test.pkl')

SEED = 0

PARAMS = {
    'fold': {
        'n_splits': 5,
        'shuffle': True,
        'random_state': SEED,
    },

    'model': {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'metric': 'auc',
        'learning_rate': 0.065,
        'num_leaves': 2**8,
        'max_depth': -1,
        'subsample_freq': 1,
        'colsample_bytree': 0.7,
        'subsample': 0.7,
        'n_estimators': 10000,
        'max_bin': 255,
        'verbose': -1,
        'n_jobs': -1,
        'seed': SEED,
    },

    'fit': {
        'early_stopping_rounds': 100,
        'eval_metric': 'auc',
        'verbose': 10,
    }

}
