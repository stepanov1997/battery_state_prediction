import os

import dill as pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from main import read_battery_data, parse_battery_data

data_path = "C:\\Users\\stepa\\PycharmProjects\\battery_state_prediction\\data"
results_path = os.path.join(data_path, "results")
specific_result_path = os.path.join(results_path, "2023-11-23-23-28-03-mlp-nn-0.0011")

estimator_path = os.path.join(specific_result_path, 'estimators\\estimator_0.0042_xgboost.pkl')
preprocessor_path = os.path.join(specific_result_path, 'preprocessor.pkl')
test_data_path = os.path.join(data_path, '5. Battery Data Set\\6. BatteryAgingARC_53_54_55_56\\B0054.mat')


def load_object(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


if __name__ == '__main__':
    estimator = load_object(estimator_path)
    preprocessor = load_object(preprocessor_path)

    X_test, y_test = preprocessor.transform([test_data_path])

    predictions = estimator.predict(X_test)

    df = pd.DataFrame(predictions, columns=["predicted"])
    df["actual"] = y_test

    print(df)
    print()

    mse = round(mean_squared_error(y_test, predictions), 5)
    print(f'MSE = {mse}')

    print()
    r2 = r2_score(y_test, predictions)
    print(f'R2 = {r2}')
