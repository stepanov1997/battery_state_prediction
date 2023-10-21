import numpy as np
import scipy.io
import pandas as pd

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
    for cycle in cycles:
        cycle_type = cycle['type']  # Vrsta ciklusa (charge, discharge, impedance)
        data = cycle['data']

        match cycle_type:
            case 'impedance':
                min_size = min(len(value) for key, value in data.items() if key not in ('Re', 'Rct'))
                data['Re'] = np.array([data['Re']] * min_size, dtype=np.float64)
                data['Rct'] = np.array([data['Rct']] * min_size, dtype=np.float64)
                for key, value in data.items():
                    data[key] = value[:min_size]

        # Kreirajte DataFrame za svaki ciklus
        cycle_df = pd.DataFrame(data)

        match cycle_type:
            case 'charge':
                cycle_df['Current_charge_load'] = cycle_df['Current_charge']
                cycle_df['Voltage_charge_load'] = cycle_df['Voltage_charge']
                del cycle_df['Current_charge']
                del cycle_df['Voltage_charge']
                cycle_df['health'] = cycle_df['Voltage_measured'] / 4.2
            case 'discharge':
                cycle_df['Current_charge_load'] = cycle_df['Current_load']
                cycle_df['Voltage_charge_load'] = cycle_df['Voltage_load']
                del cycle_df['Current_load']
                del cycle_df['Voltage_load']
                cycle_df['health'] = cycle_df['Voltage_measured'] / 4.2


        # Dodajte kolonu koja označava vrstu ciklusa
        cycle_df['cycle_type'] = cycle_type


        # Dodajte podatke iz ovog ciklusa u glavni DataFrame
        df = pd.concat([df, cycle_df], ignore_index=True)

print(df)

df.to_csv('experiment1_dataset_v1.csv')

# Sada imate DataFrame df koji sadrži sve podatke iz mat fajlova
# Možete dalje raditi s ovim DataFrame-om za treniranje modela za duboko učenje.
