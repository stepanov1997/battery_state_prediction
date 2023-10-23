import numpy as np
import scipy.io
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

        df = pd.concat([df, cycle_df], ignore_index=True)

    df = df.round(3)

    return df


# noinspection PyShadowingNames
def split_dataframe(df):
    # X will contain n-1 columns, i.e., all except the last one
    X = df.iloc[:, :-1]

    # y will contain the last column
    y = df.iloc[:, -1]

    # First, split the data into a training set (70%) and a temporary set (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

    # Then, split the temporary set into a validation set (20%) and a test set (10%)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.333, random_state=42)

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


if __name__ == '__main__':
    dataframe = create_dataframe_with_data(DATA_DIRECTORY_FILENAME)
    print(dataframe)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataframe(dataframe)

    plot_dataset(X_train['time'], y_train, X_train['Temperature_measured_max'], label='Training Data')
    plot_dataset(X_val['time'], y_val, X_val['Temperature_measured_max'], label='Validation Data')
    plot_dataset(X_test['time'], y_test, X_test['Temperature_measured_max'], label='Test Data')

    dataframe.to_csv('experiment1_dataset_v1.csv', index=False)
