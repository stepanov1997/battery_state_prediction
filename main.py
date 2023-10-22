import numpy as np
import scipy.io
import pandas as pd

if __name__=='__main__':
    # Lista baterija
    battery_numbers = [5, 6, 7, 18]

    # Prazan DataFrame koji će sadržati sve podatke
    df = pd.DataFrame()

    # Iterirajte kroz svaku bateriju i učitajte podatke
    for battery_number in battery_numbers[:1]:
        battery_name = f'B00{battery_number:02d}'
        filename = f'C:\\Users\\stepa\\PycharmProjects\\battery_state_prediction\\data\\5. Battery Data Set\\1. BatteryAgingARC-FY08Q4\\{battery_name}.mat'
        mat_data = scipy.io.loadmat(filename, simplify_cells=True)

        # Prikupite cikluse za svaku bateriju (charge, discharge, impedance)
        battery_data = mat_data[battery_name]
        cycles = battery_data['cycle']

        # Iterirajte kroz sve cikluse
        cycles = [cycle for cycle in cycles if cycle['type'] == 'charge']

        for cycle in cycles:
            voltage_measured = cycle['data']['Voltage_measured']
            cycle['Voltage_measured_min'] = min(voltage_measured)
            cycle['Voltage_measured_max'] = max(voltage_measured)

            current_measured = cycle['data']['Current_measured']
            cycle['Current_measured_min'] = min(current_measured)
            cycle['Current_measured_max'] = max(current_measured)

            temperature_measured = cycle['data']['Temperature_measured']
            cycle['Temperature_measured_min'] = min(temperature_measured)
            cycle['Temperature_measured_max'] = max(temperature_measured)

            current_charge = cycle['data']['Current_charge']
            cycle['Current_charge_min'] = min(current_charge)
            cycle['Current_charge_max'] = max(current_charge)

            voltage_charge = cycle['data']['Voltage_charge']
            cycle['Voltage_charge_min'] = min(voltage_charge)
            cycle['Voltage_charge_max'] = max(voltage_charge)

            time = cycle['data']['Time']
            cycle['Time_max'] = max(time)

            cycle['health'] = max(voltage_measured) / 4.2

            del cycle['data']

        # Kreirajte DataFrame za svaki ciklus
        cycle_df = pd.DataFrame(cycles)


        # Dodajte podatke iz ovog ciklusa u glavni DataFrame
        df = pd.concat([df, cycle_df], ignore_index=True)

    print(df)

    df.to_csv('experiment1_dataset_v1.csv')

    # Sada imate DataFrame df koji sadrži sve podatke iz mat fajlova
    # Možete dalje raditi s ovim DataFrame-om za treniranje modela za duboko učenje.