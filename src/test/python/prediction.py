import os

import dill as pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.utils import plot_model
import sys

PROJECT_PATH = "C:\\Users\\stepa\\PycharmProjects\\battery_state_prediction"

sys.path.append(f"{PROJECT_PATH}\\src\\main\\python")

data_path = f"{PROJECT_PATH}\\data"
results_path = os.path.join(data_path, "results")
specific_result_path = os.path.join(results_path, "2023-11-24-03-18-35-mlp-nn-0.0013")

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

    if isinstance(estimator[2], KerasRegressor):
        estimator[2].model_.summary()
        plot_model(estimator[2].model_, to_file='model.png', show_shapes=True, show_layer_names=True,
                   show_trainable=True, show_layer_activations=True)
        results = permutation_importance(estimator, X_test, y_test, scoring='neg_mean_squared_error', random_state=42)
        # get importance
        feature_importances = results.importances_mean[1:]
        print(len(feature_importances))
    else:
        feature_importances = estimator[2].feature_importances_
        print(len(feature_importances))

    importance_dict = dict(zip(X_test.columns[1:], feature_importances))
    plt.figure(figsize=(10, 6))
    plt.bar(importance_dict.keys(), importance_dict.values(), color='skyblue')
    plt.xticks(rotation=90)  # Rotate labels for better readability
    plt.xlabel('Parameters')
    plt.ylabel('Importance (Log Scale)')
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.title('Importance of Different Parameters in Battery Health Prediction (Log Scale)')
    plt.tight_layout()  # Adjust layout for better display
    plt.show()

    mse = round(mean_squared_error(y_test, predictions), 5)
    print(f'MSE = {mse}')

    print()
    r2 = r2_score(y_test, predictions)
    print(f'R2 = {r2}')
