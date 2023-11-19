import numpy as np
import scipy.io
import pandas as pd
import os
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten
from joblib import Memory
from functools import lru_cache
import json
from scipy.stats import kurtosis
from scikeras.wrappers import KerasRegressor
from datetime import datetime
import dill as pickle
import time

DATA_DIRECTORY_FILENAME = 'C:\\Users\\stepa\\PycharmProjects\\battery_state_prediction\\data'

memory = Memory(location=f'{DATA_DIRECTORY_FILENAME}\\.cache', verbose=0)


def read_data(file):
    battery_name = os.path.splitext(os.path.basename(file))[0]
    battery_data = scipy.io.loadmat(file, simplify_cells=True)
    return battery_data[battery_name]


@lru_cache
def parse_battery_data(file):
    battery_df = pd.DataFrame(read_data(file))
    battery_df['battery_filename'] = file

    first_level_data = battery_df['cycle'].apply(pd.Series)
    second_level_data = first_level_data['data'].apply(pd.Series)

    battery_df = battery_df.join(first_level_data) \
        .join(second_level_data)
    battery_df = battery_df[(battery_df['type'] == 'discharge') & (battery_df['Capacity'].notna())]
    return battery_df.drop(['cycle', 'data', 'type'], axis=1) \
        .dropna(axis=1, how='all') \
        .reset_index(drop=True)


def read_and_parse_files(files):
    battery_dfs = [parse_battery_data(file) for file in files]
    return pd.concat(battery_dfs, ignore_index=True)


def create_result_folders():
    main_folder = os.path.join(DATA_DIRECTORY_FILENAME, 'results')
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)

    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    subfolder = os.path.join(main_folder, timestamp)
    os.makedirs(subfolder)

    return subfolder, timestamp


def serialize_preprocessor(directory, preprocessor):
    with open(os.path.join(directory, 'preprocessor.pkl'), 'wb') as file:
        pickle.dump(preprocessor, file)


def serialize_estimator(subfolder, name, estimator, statistics):
    estimators_subfolder = os.path.join(subfolder, "estimators")

    if not os.path.exists(estimators_subfolder):
        os.makedirs(estimators_subfolder)

    estimator_filename = f'estimator_{statistics["mse"]}_{name}'
    with open(os.path.join(estimators_subfolder, f'{estimator_filename}.pkl'), 'wb') as file:
        pickle.dump(estimator, file)

    with open(os.path.join(estimators_subfolder, f'{estimator_filename}_statistics.json'), 'w') as file:
        json.dump(statistics, file)


def serialize_results(subfolder, results_df):
    result_json = json.loads(results_df.to_json())

    with open(os.path.join(subfolder, "results.json"), "w") as f:
        json.dump(result_json, f, indent=4)


def plot_and_save_charts(subfolder, results_df):
    # Kreiranje DataFrame-a
    df_fit_duration = pd.DataFrame({'model': results_df.index, 'fit_duration': results_df['fit_duration']})
    df_prediction_duration = pd.DataFrame(
        {'model': results_df.index, 'prediction_duration': results_df['prediction_duration']})
    df_mse = pd.DataFrame({'model': results_df.index, 'mse': results_df['mse']})
    df_r2 = pd.DataFrame({'model': results_df.index, 'r2': results_df['r2']})

    def plot_and_save_chart(df, y_label, file_name):
        plt.figure(figsize=(10, 6))
        plt.plot(df['model'], df.iloc[:, 1], marker='o')
        plt.xticks(rotation=45)
        plt.ylabel(y_label)
        plt.title(f'{y_label} by model')

        # Čuvanje grafikona kao slike
        plt.savefig(os.path.join(subfolder, f"{file_name}.png"))
        plt.show()

    # Plotovanje grafikona i čuvanje kao slike
    plot_and_save_chart(df_fit_duration, 'Fit Duration', 'fit_duration_chart')
    plot_and_save_chart(df_prediction_duration, 'Prediction Duration', 'prediction_duration_chart')
    plot_and_save_chart(df_mse, 'MSE', 'mse_chart')
    plot_and_save_chart(df_r2, 'R2', 'r2_chart')


def prefit_preprocessing(df):
    def describe_nested_data(column_name):
        df[f'{column_name}_max'] = df[column_name].apply(np.max)
        df[f'{column_name}_min'] = df[column_name].apply(np.min)
        df[f'{column_name}_avg'] = df[column_name].apply(np.average)
        df[f'{column_name}_std'] = df[column_name].apply(np.std)
        # df[f'{column_name}_kurt'] = df[column_name].apply(kurtosis)
        df.drop([column_name], axis=1, inplace=True)

    for column_name in ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'Current_load', 'Voltage_load']:
        describe_nested_data(column_name)

    df['Time_max'] = df['Time'].apply(np.max)
    df = df.drop(['Time', 'time'], axis=1) \
        .round(5)

    y = pd.to_numeric(df['Capacity'] / 2, errors='coerce').apply(lambda health: 1 if health >= 1 else health)

    X = df.drop(['Capacity'], axis=1) \
        .drop(y[y.isna()].index) \
        .dropna()

    y = y.drop(y[y.isna()].index)

    return X, y


# Add this function to create a neural network model
def create_mlp_nn_model(input_shape):
    def f(optimizer='adam', neurons_layer_1=32, neurons_layer_2=16, activation='relu'):
        model = Sequential()
        model.add(Dense(neurons_layer_1, activation=activation, input_shape=(input_shape,)))
        model.add(Dense(neurons_layer_2, activation=activation))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model

    return f


def create_lstm_nn_model(input_shape):
    def f(lstm_units=20, dense_units=10, activation='relu', optimizer='adam'):
        model = Sequential()

        model.add(LSTM(lstm_units, activation=activation, input_shape=(input_shape, 1)))
        model.add(Dense(dense_units, activation=activation))
        model.add(Dense(1, activation='linear'))

        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model

    return f


def create_cnn_model(input_shape):
    def f(filters=32, kernel_size=3, dense_units=10, activation='relu', optimizer='adam'):
        model = Sequential()

        model.add(Conv1D(filters, kernel_size=kernel_size, activation=activation, input_shape=(input_shape, 1)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(dense_units, activation=activation))
        model.add(Dense(1, activation='linear'))

        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model

    return f


def train_model_with_estimator(X_train, y_train, estimator_tuple, grid_params):
    groups = X_train['battery_filename']

    pipeline = Pipeline([
        ('small_modifier', FunctionTransformer(func=lambda x: x.drop(['battery_filename'], axis=1).astype(np.float64))),
        ('scaler', StandardScaler()),
        estimator_tuple
    ])

    group_kfold = GroupKFold(n_splits=10)
    grid_search = GridSearchCV(pipeline, grid_params, verbose=2, cv=group_kfold, scoring='neg_mean_squared_error',
                               n_jobs=-1)
    grid_search.fit(X_train, y_train.astype(np.float64), groups=groups)

    return grid_search


def calculate_errors_on_test_set(grid_search, X_test, y_test):
    y_pred = grid_search.predict(X_test)

    mse = round(mean_squared_error(y_test, y_pred), 4)
    r2 = r2_score(y_test, y_pred)

    return mse, r2


def train_model(train_folder, X_train, y_train, X_test, y_test, estimators_data):
    best_global_estimator_name = None
    best_global_mse = 1.0
    best_global_r2 = 0.0
    best_global_estimator = None
    results = {}
    estimators = [(estimator_dict['estimator'], estimator_dict['grid_param']) for estimator_dict in estimators_data]
    for estimator, grid_param in estimators:
        fit_start_time = time.perf_counter()
        grid_search = train_model_with_estimator(X_train, y_train, estimator, grid_param)
        fit_duration = time.perf_counter() - fit_start_time

        prediction_start_time = time.perf_counter()
        best_local_mse, best_local_r2 = calculate_errors_on_test_set(grid_search, X_test, y_test)
        prediction_duration = time.perf_counter() - prediction_start_time

        best_local_estimator = grid_search.best_estimator_
        estimator_name = estimator[0]
        statistics = {
            'name': estimator_name,
            'best_params': grid_search.best_params_,
            'fit_duration': fit_duration,
            'prediction_duration': prediction_duration,
            'mse': best_local_mse,
            'r2': best_local_r2,
        }
        serialize_estimator(train_folder, estimator_name, best_local_estimator, statistics)
        results[estimator_name] = statistics
        if best_local_mse >= best_global_mse:
            continue
        best_global_mse = best_local_mse
        best_global_r2 = best_local_r2
        best_global_estimator_name = estimator_name
        best_global_estimator = best_local_estimator

    return results, best_global_estimator_name, best_global_estimator, best_global_mse, best_global_r2


if __name__ == '__main__':
    battery_filenames = pd.Series([
        f'{root}\\{filename}'
        for root, _, filenames in os.walk(DATA_DIRECTORY_FILENAME)
        for filename in filenames
        if filename.startswith('B00') and filename.endswith('.mat')
    ])

    train, test = train_test_split(battery_filenames, test_size=0.2, random_state=42)

    # Fit preprocessing pipeline on training data
    preprocessing_pipeline = Pipeline([
        ('read_and_parse_files', FunctionTransformer(func=read_and_parse_files)),
        ('prefit_preprocessing', FunctionTransformer(func=prefit_preprocessing))
    ])

    # Transform both training and test data using the fitted pipeline
    X_train, y_train = preprocessing_pipeline.fit_transform(train)
    X_test, y_test = preprocessing_pipeline.transform(test)

    input_shape = X_train.shape[1] - 1

    train_folder, timestamp = create_result_folders()
    serialize_preprocessor(train_folder, preprocessing_pipeline)

    estimators_data = [
        {
            'estimator': ('linear', LinearRegression(n_jobs=-1)),
            'grid_param': {
                'scaler__with_std': [True, False],
                # 'linear__fit_intercept': [True, False],
                # 'linear__positive': [True, False]
            }
        },
        {
            'estimator': ('ridge', Ridge()),
            'grid_param': {
                'scaler__with_std': [True, False],
                # 'ridge__alpha': [0.1, 1.0, 10.0, 100.0],
                # 'ridge__fit_intercept': [True, False],
                # 'ridge__tol': [1e-4, 1e-3, 1e-2],
                # 'ridge__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
            }
        },
        {
            'estimator': ('lasso', Lasso()),
            'grid_param': {
                'scaler__with_std': [True, False],
                # 'lasso__alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10],
                # 'lasso__fit_intercept': [True, False],
                # 'lasso__tol': [1e-4, 1e-3, 1e-2],
                # 'lasso__selection': ['cyclic', 'random'],
            }
        },
        {
            'estimator': ('random_forest', RandomForestRegressor(n_jobs=-1)),
            'grid_param': {
                'scaler__with_std': [True, False],
                # 'random_forest__n_estimators': [1, 3, 5],
                # 'random_forest__max_depth': [None, 3, 5, 10],
                # 'random_forest__min_samples_split': [2, 4, 6],
                # 'random_forest__min_samples_leaf': [1, 2, 4],
                # 'random_forest__max_features': ['sqrt', 'log2'],
                # 'random_forest__bootstrap': [True, False],
            }
        },
        {
            'estimator': ('xgboost', XGBRegressor(n_jobs=-1)),
            'grid_param': {
                'scaler__with_std': [True, False],
                # 'xgboost__n_estimators': [50, 100, 300, 500],
                # 'xgboost__max_depth': [3, 4, 5, 6],
                # 'xgboost__learning_rate': [0.01, 0.1, 0.2, 0.3],
                # 'xgboost__min_child_weight': [1, 2, 3, 4],
                # 'xgboost__gamma': [0, 0.1, 0.2, 0.3, 0.4],
                # 'xgboost__subsample': [0.5, 0.75, 1],
                # 'xgboost__colsample_bytree': [0.5, 0.75, 1],
                # 'xgboost__reg_alpha': [0, 0.1, 0.2, 0.5, 1],
                # 'xgboost__reg_lambda': [0, 0.1, 0.5, 1]
            }
        },
        {
            'estimator': ('svm', SVR(max_iter=1000)),
            'grid_param': {
                'scaler__with_std': [True, False],
                # 'svm__C': [0.1, 1.0, 10.0],  # Parametar regularizacije
                # 'svm__kernel': ['linear', 'rbf', 'poly'],  # Tip jezgra
                # 'svm__gamma': ['scale', 'auto', 0.1, 1, 10],  # Koeficijent za 'rbf', 'poly' i 'sigmoid'
                # 'svm__degree': [1, 2, 3, 4, 5],  # Stepen za 'poly' jezgro
                # 'svm__coef0': [0.0, 0.1, 0.5, 1],  # Nezavisni termin u kernel funkciji
                # 'svm__epsilon': [0.1, 0.2, 0.5, 1]  # Epsilon u epsilon-SVR modelu
            }
        },
        # Neural networks
        {
            'estimator': ('mlp-nn', KerasRegressor(model=create_mlp_nn_model(input_shape))),
            'grid_param': {
                'scaler__with_std': [True, False],
                # 'mlp-nn__epochs': [50, 100, 150],
                # 'mlp-nn__batch_size': [64, 128, 256],
                # 'mlp-nn__model__neurons_layer_1': [32, 64, 128],
                # 'mlp-nn__model__neurons_layer_2': [16, 32, 64],
                # 'mlp-nn__model__activation': ['relu', 'tanh', 'sigmoid'],
                'mlp-nn__model__optimizer': ['rmsprop', 'adam'],
            }
        },
        {
            'estimator': ('lstm-nn', KerasRegressor(model=create_lstm_nn_model(input_shape))),
            'grid_param': {
                'scaler__with_std': [True, False],
                # 'lstm-nn__epochs': [50, 100, 150],
                # 'lstm-nn__batch_size': [64, 128, 256],
                # 'lstm-nn__model__lstm_units': [20, 50, 100],
                # 'lstm-nn__model__dense_units': [10, 20, 50],
                # 'lstm-nn__model__activation': ['relu', 'tanh', 'sigmoid'],
                'lstm-nn__model__optimizer': ['rmsprop', 'adam']
            }
        },
        {
            'estimator': ('cnn-nn', KerasRegressor(model=create_cnn_model(input_shape))),
            'grid_param': {
                'scaler__with_std': [True, False],
                # 'cnn-nn__epochs': [50, 100, 150],
                # 'cnn-nn__batch_size': [64, 128, 256],
                # 'cnn-nn__model__filters': [32, 64, 128],
                # 'cnn-nn__model__kernel_size': [2, 3, 5],
                # 'cnn-nn__model__dense_units': [10, 20, 50],
                # 'cnn-nn__model__activation': ['relu', 'tanh', 'sigmoid'],
                'cnn-nn__model__optimizer': ['rmsprop', 'adam']
            }
        }
    ]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost")
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")

        results, best_global_estimator_name, best_global_estimator, best_global_mse, best_global_r2 = train_model(
            train_folder, X_train, y_train, X_test, y_test, estimators_data
        )
        old_folder = train_folder
        train_folder = f"{old_folder}-{best_global_estimator_name}-{best_global_mse}"
        os.rename(old_folder, train_folder)

    print()
    print(f'Best estimator: {best_global_estimator_name}')
    print(f'Best MSE: {best_global_mse:0.5f} %')
    print(f'Best R2: {best_global_r2:0.5f} %')

    results_df = pd.DataFrame.from_dict(results, orient='index') \
        .drop(['name'], axis=1) \
        .sort_values(by='mse')

    serialize_results(train_folder, results_df)
    plot_and_save_charts(train_folder, results_df)
