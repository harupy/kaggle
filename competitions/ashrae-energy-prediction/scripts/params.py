SEED = 42
PARAMS = {
    'model': {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'n_estimators': 10000,
        'learning_rate': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': SEED
    },
    'fit': {
        'eval_metric': 'rmse',
        'early_stopping_rounds': 100,
        'verbose': 100,
    },
    'fold': {
        'n_splits': 5,
        'shuffle': True,
        'random_state': SEED,
    },
}
