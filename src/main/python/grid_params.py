from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from scikeras.wrappers import KerasRegressor
from neural_network_generator import NeuralNetworkGenerator


def load_estimators_data(input_shape):
    return [
        {
            'estimator': ('linear', LinearRegression(n_jobs=-1)),
            'grid_param': {
                # 'scaler__with_std': [True, False],
                # 'linear__fit_intercept': [True, False],
                # 'linear__positive': [True, False]

                # Best results
                "linear__fit_intercept": [True],
                "linear__positive": [True],
                "scaler__with_std": [False]
            }
        },
        {
            'estimator': ('ridge', Ridge()),
            'grid_param': {
                # 'scaler__with_std': [True, False],
                # 'ridge__alpha': [0.1, 1.0, 10.0, 100.0],
                # 'ridge__fit_intercept': [True, False],
                # 'ridge__tol': [1e-4, 1e-3, 1e-2],
                # 'ridge__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']

                # Best results
                "scaler__with_std": [False],
                "ridge__alpha": [10.0],
                "ridge__fit_intercept": [True],
                "ridge__tol": [0.0001],
                "ridge__solver": ["svd"],
            }
        },
        {
            'estimator': ('lasso', Lasso()),
            'grid_param': {
                # 'scaler__with_std': [True, False],
                # 'lasso__alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10],
                # 'lasso__fit_intercept': [True, False],
                # 'lasso__tol': [1e-4, 1e-3, 1e-2],
                # 'lasso__selection': ['cyclic', 'random'],

                # Best results
                "scaler__with_std": [True],
                "lasso__alpha": [0.001],
                "lasso__fit_intercept": [True],
                "lasso__tol": [0.01],
                "lasso__selection": ["cyclic"],
            }
        },
        {
            'estimator': ('random_forest', RandomForestRegressor(n_jobs=-1)),
            'grid_param': {
                # 'scaler__with_std': [True, False],
                # 'random_forest__n_estimators': [1, 3, 5],
                # 'random_forest__max_depth': [None, 3, 5, 10],
                # 'random_forest__min_samples_split': [2, 4, 6],
                # 'random_forest__min_samples_leaf': [1, 2, 4],
                # 'random_forest__max_features': ['sqrt', 'log2'],
                # 'random_forest__bootstrap': [True, False],

                # Best results
                "scaler__with_std": [False],
                "random_forest__n_estimators": [5],
                "random_forest__max_depth": [None],
                "random_forest__min_samples_split": [4],
                "random_forest__min_samples_leaf": [1],
                "random_forest__max_features": ["sqrt"],
                "random_forest__bootstrap": [True],
            }
        },
        {
            'estimator': ('catboost', CatBoostRegressor()),
            'grid_param': {
                # 'scaler__with_std': [True, False],
                # 'catboost__depth': [4, 5, 6],
                # 'catboost__learning_rate': [0.01, 0.1, 0.2],
                # 'catboost__l2_leaf_reg': [0.1, 0.2, 0.3],
                # 'catboost__min_child_samples': [100, 200, 300],
                # 'catboost__subsample': [0.5, 0.75, 1],
                # 'catboost__colsample_bylevel': [0.5, 0.75, 1],
                # 'catboost__loss_function': ["RMSE", "MAE", "Quantile:alpha=0.5"],
                # 'catboost__bootstrap_type': ["Bayesian", "Bernoulli", "MVS"],
                # 'catboost__eval_metric': ["RMSE", "MAE", "R2"]

                # Best results
                "scaler__with_std": [False],
                "catboost__depth": [4],
                "catboost__learning_rate": [0.1],
                "catboost__l2_leaf_reg": [0.1],
                "catboost__min_child_samples": [100],
                "catboost__subsample": [0.75],
                "catboost__colsample_bylevel": [0.75],
                "catboost__loss_function": ["RMSE"],
                "catboost__bootstrap_type": ["Bernoulli"],
                "catboost__eval_metric": ["RMSE"],
            }
        },
        {
            'estimator': ('xgboost', XGBRegressor(n_jobs=-1)),
            'grid_param': {
                # 'scaler__with_std': [True, False],
                # 'xgboost__n_estimators': [50, 100, 300],
                # 'xgboost__max_depth': [4, 5, 6],
                # 'xgboost__learning_rate': [0.01, 0.1, 0.2],
                # 'xgboost__min_child_weight': [1, 2, 3],
                # 'xgboost__gamma': [0, 0.1, 0.2, 0.3],
                # 'xgboost__subsample': [0.5, 0.75, 1],
                # 'xgboost__colsample_bytree': [0.5, 0.75, 1],
                # 'xgboost__reg_alpha': [0, 0.1, 0.5],
                # 'xgboost__reg_lambda': [0, 0.1, 0.5]

                # Best results
                "scaler__with_std": [True],
                "xgboost__n_estimators": [300],
                "xgboost__max_depth": [6],
                "xgboost__learning_rate": [0.1],
                "xgboost__min_child_weight": [2],
                "xgboost__gamma": [0],
                "xgboost__subsample": [0.75],
                "xgboost__colsample_bytree": [0.5],
                "xgboost__reg_alpha": [0],
                "xgboost__reg_lambda": [0.5],
            }
        },
        {
            'estimator': ('svm', SVR(max_iter=1000)),
            'grid_param': {
                # 'scaler__with_std': [True, False],
                # 'svm__C': [0.1, 1.0, 10.0],
                # 'svm__kernel': ['linear', 'rbf', 'poly'],
                # 'svm__gamma': ['scale', 'auto', 0.1, 1, 10],
                # 'svm__degree': [1, 3, 5],
                # 'svm__coef0': [0.0, 0.1, 0.5],
                # 'svm__epsilon': [0.1, 0.2, 0.5]

                # Best results
                'scaler__with_std': [False],
                "svm__C": [10.0],
                "svm__kernel": ["rbf"],
                "svm__gamma": ["scale"],
                "svm__degree": [1],
                "svm__coef0": [0.0],
                "svm__epsilon": [0.1],
            }
        },
        # Neural networks
        {
            'estimator': ('mlp-nn', KerasRegressor(model=NeuralNetworkGenerator.generate_mlp_model(input_shape))),
            'grid_param': {
                # 'scaler__with_std': [True, False],
                # 'mlp-nn__epochs': [50],
                # 'mlp-nn__batch_size': [128],
                # 'mlp-nn__model__neurons_layer_1': [32, 64, 128],
                # 'mlp-nn__model__neurons_layer_2': [16, 32, 64],
                # 'mlp-nn__model__activation': ['relu', 'tanh', 'sigmoid'],
                # 'mlp-nn__model__optimizer': ['rmsprop', 'adam'],

                # Best results
                'scaler__with_std': [True],
                "mlp-nn__epochs": [50],
                "mlp-nn__batch_size": [128],
                "mlp-nn__model__neurons_layer_1": [64],
                "mlp-nn__model__neurons_layer_2": [64],
                "mlp-nn__model__activation": ["tanh"],
                "mlp-nn__model__optimizer": ["adam"],
            }
        },
        {
            'estimator': ('lstm-nn', KerasRegressor(model=NeuralNetworkGenerator.generate_lstm_model(input_shape))),
            'grid_param': {
                # 'scaler__with_std': [True, False],
                # 'lstm-nn__epochs': [50],
                # 'lstm-nn__batch_size': [128],
                # 'lstm-nn__model__lstm_units': [20, 50, 100],
                # 'lstm-nn__model__dense_units': [10, 20, 50],
                # 'lstm-nn__model__activation': ['relu', 'tanh', 'sigmoid'],
                # 'lstm-nn__model__optimizer': ['rmsprop', 'adam']

                # Best results
                "scaler__with_std": [False],
                "lstm-nn__epochs": [50],
                "lstm-nn__batch_size": [128],
                "lstm-nn__model__lstm_units": [20],
                "lstm-nn__model__dense_units": [10],
                "lstm-nn__model__activation": ["tanh"],
                "lstm-nn__model__optimizer": ["rmsprop"],
            }
        },
        {
            'estimator': ('cnn-nn', KerasRegressor(model=NeuralNetworkGenerator.create_cnn_model(input_shape))),
            'grid_param': {
                # 'scaler__with_std': [True, False],
                # 'cnn-nn__epochs': [50],
                # 'cnn-nn__batch_size': [128],
                # 'cnn-nn__model__filters': [32, 64, 128],
                # 'cnn-nn__model__kernel_size': [2, 3, 5],
                # 'cnn-nn__model__dense_units': [10, 20, 50],
                # 'cnn-nn__model__activation': ['relu', 'tanh', 'sigmoid'],
                # 'cnn-nn__model__optimizer': ['rmsprop', 'adam']

                # Best results
                "scaler__with_std": [False],
                "cnn-nn__epochs": [50],
                "cnn-nn__batch_size": [128],
                "cnn-nn__model__filters": [128],
                "cnn-nn__model__kernel_size": [3],
                "cnn-nn__model__dense_units": [10],
                "cnn-nn__model__activation": ["tanh"],
                "cnn-nn__model__optimizer": ["adam"],
            }
        }
    ]
