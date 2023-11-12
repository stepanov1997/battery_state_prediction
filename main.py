import numpy as np
import scipy.io
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import warnings

DATA_DIRECTORY_FILENAME = 'C:\\Users\\stepa\\PycharmProjects\\battery_state_prediction\\data'


# noinspection PyShadowingNames
def create_dataframe_with_data(data_directory_filename):
    df = pd.DataFrame()

    battery_filenames = [
        f'{root}\\{filename}'
        for root, _, filenames in os.walk(data_directory_filename)
        for filename in filenames
        if filename.startswith('B00') and filename.endswith('.mat')
    ]
    print(battery_filenames)

    for battery_filename in battery_filenames:
        mat_data = scipy.io.loadmat(battery_filename, simplify_cells=True)

        battery_data = mat_data[os.path.splitext(os.path.basename(battery_filename))[0]]
        cycles = battery_data['cycle']

        cycles = [cycle for cycle in cycles if cycle['type'] == 'discharge']

        for cycle in cycles:
            voltage_measured = cycle['data']['Voltage_measured']
            cycle['Voltage_measured_max'] = max(voltage_measured)

            current_measured = cycle['data']['Current_measured']
            cycle['Current_measured_min'] = min(current_measured)
            cycle['Current_measured_max'] = max(current_measured)

            temperature_measured = cycle['data']['Temperature_measured']
            cycle['Temperature_measured_min'] = min(temperature_measured)
            cycle['Temperature_measured_max'] = max(temperature_measured)

            current_charge = cycle['data'].get('Current_charge', cycle['data']['Current_load'])
            cycle['Current_charge_min'] = min(current_charge)
            cycle['Current_charge_max'] = max(current_charge)

            voltage_charge = cycle['data'].get('Voltage_charge', cycle['data']['Voltage_load'])
            cycle['Voltage_charge_min'] = min(voltage_charge)
            cycle['Voltage_charge_max'] = max(voltage_charge)

            time = cycle['data']['Time']
            cycle['time'] = max(time)

            cycle['health'] = cycle['data']['Capacity'] / 2.0

            del cycle['data']
            del cycle['type']

        cycle_df = pd.DataFrame(cycles)

        cycle_df['health'] = pd.to_numeric(cycle_df['health'], errors='coerce').apply(lambda x: 1 if x >= 1 else x)

        cycle_df = cycle_df.dropna()

        df = pd.concat([df, cycle_df], ignore_index=True)

    df = df.round(3)

    return df


# noinspection PyShadowingNames
def split_dataframe(X, y):
    # First, split the data into a training set (70%) and a temporary set (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

    # Then, split the temporary set into a validation set (10%) and a test set (20%)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.666, random_state=42)

    # Now you have X_train, y_train for training, X_val, y_val for validation, and X_test, y_test for testing.
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def plot_dataset(X, y, c, label):
    plt.scatter(X, y, c=c, label='Temperature [°C]')
    plt.colorbar()
    plt.xlabel('Discharging time [s]')
    plt.ylabel('SOH (state of health)')
    plt.title(f'Scatter Plot of {label}')
    plt.legend()
    plt.show()


def scale_dataset(X):
    column_names = X.columns
    scaler = StandardScaler()
    # Fit the scaler to the data and transform the data
    scaled_data = scaler.fit_transform(X)
    return pd.DataFrame(scaled_data, columns=column_names), scaler


def select_features_in_dataset(X, y):
    selector = SelectKBest(score_func=f_classif, k=5)
    selected_features = selector.fit_transform(X, y)
    cols_idxs = selector.get_support(indices=True)
    return pd.DataFrame(selected_features, columns=X.columns[cols_idxs]), selector


def train_model_with_estimator(X_tuple, y_tuple, estimator, grid_params):
    (X_train, X_val, X_test), (y_train, y_val, y_test) = X_tuple, y_tuple
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        estimator
    ])

    grid_search = GridSearchCV(pipeline, grid_params, verbose=1, cv=10, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    y_true = pd.concat([y_val, y_test]).values

    y_pred = grid_search.predict(pd.concat([X_val, X_test]))
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return grid_search.best_params_, mse, r2


def train_model(X_tuple, y_tuple, estimators_data):
    best_mse = 1.0
    best_r2 = 0.0
    best_estimator = None
    estimators = [(estimator_dict['estimator'], estimator_dict['grid_param']) for estimator_dict in estimators_data]
    for estimator, grid_param in estimators:
        best_params, mse, r2 = train_model_with_estimator(X_tuple, y_tuple, estimator, grid_param)
        if mse < best_mse:
            best_mse = mse
            best_r2 = r2
            best_estimator = (estimator, best_params)

    return best_estimator, best_mse, best_r2


if __name__ == '__main__':
    dataframe = create_dataframe_with_data(DATA_DIRECTORY_FILENAME)
    print(dataframe)

    X = dataframe.iloc[:, :-1]
    y = dataframe.iloc[:, -1]

    # X, scaler = scale_dataset(X)
    # X, selector = select_features_in_dataset(X, y)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataframe(X, y)

    plot_dataset(X_train['Temperature_measured_max'], y_train, X_train['ambient_temperature'], label='Training Data')

    dataframe.to_csv('experiment1_dataset_v1.csv', index=False)

    estimators_data = [
        {
            'estimator': ('linear', LinearRegression()),
            'grid_param': {
                'scaler__with_std': [True, False],
                'linear__fit_intercept': [True, False],
                'linear__n_jobs': [1, 3, 5, 8],
                'linear__positive': [True, False],
            }
        },
        {
            'estimator': ('ridge', Ridge()),
            'grid_param': {
                'scaler__with_std': [True, False],
                'ridge__alpha': [0.1, 1.0, 10.0],
            }
        },
        {
            'estimator': ('lasso', Lasso()),
            'grid_param': {
                'scaler__with_std': [True, False],
                'lasso__alpha': [0.1, 1.0, 10.0],
            }
        },
        {
            'estimator': ('random_forest', RandomForestRegressor()),
            'grid_param': {
                'random_forest__n_estimators': [1, 3, 5],
                'random_forest__max_depth': [None, 1, 3, 5],
            }
        },
        {
            'estimator': ('xgboost', XGBRegressor()),
            'grid_param': {
                'xgboost__n_estimators': [50, 100, 300, 500],
                'xgboost__max_depth': [3, 4, 5, 6],
                'xgboost__learning_rate': [0.01, 0.1, 0.2, 0.3],
            }
        },
        # {
        #     'estimator': ('svm', SVR()),
        #     'grid_param': {
        #         'scaler__with_std': [True, False],
        #         'svm__C': [0.1, 1.0, 10.0],
        #         'svm__kernel': ['linear', 'rbf', 'poly'],
        #     }
        # }
    ]

    # Suppress the FutureWarning related to is_sparse
    warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost")

    best_estimator, best_mse, best_r2 = train_model((X_train, X_val, X_test), (y_train, y_val, y_test), estimators_data)

    print()
    print(f'Best estimator: {best_estimator}')
    print(f'Best MSE: {best_mse:0.5f} %')
    print(f'Best R2: {best_r2:0.5f} %')
