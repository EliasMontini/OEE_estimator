import pandas as pd
import os
import data_analysis_methods
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import IsolationForest

def data_frame_init(file_path):
    df = pd.read_csv(file_path)
    df = df[['timestamp', 'temp', 'coppia']]
    print(df)
    return df

def data_frames_combination(folder):
    data_folder_path = folder
    dataframes = []
    for filename in os.listdir(data_folder_path):
        if filename.endswith('.csv'):
            csv_path = os.path.join(data_folder_path, filename)
            df = pd.read_csv(csv_path)
            # Add a new column 'ID' with the filename (excluding the '.csv' extension)
            df['ID'] = filename[:-4]
            dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df = combined_df[['ID', 'timestamp', 'temp', 'coppia']]
    unique_ids = combined_df['ID'].unique()
    print(unique_ids)
    print(combined_df)
    return combined_df

def plot(df, feature):
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df[feature], label=feature, linestyle='dashed')
    plt.xlabel('Index')
    plt.ylabel(feature)
    #plt.title('XXX')
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_time_delta(df, range, time_unit='minutes'):
    # Convert timestamp values to datetime objects
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Select timestamp values at the provided indices
    start_time = df.at[range[0], 'timestamp']
    end_time = df.at[range[1], 'timestamp']

    # Calculate time delta
    time_delta = end_time - start_time

    # Convert time delta to the specified unit
    if time_unit == 'seconds':
        return time_delta.total_seconds()
    elif time_unit == 'minutes':
        return time_delta.total_seconds() / 60
    elif time_unit == 'hours':
        return time_delta.total_seconds() / 3600
    else:
        return time_delta


def delete_rows_and_reset_index(df, range_vector):
    # Create a mask to filter out rows between index_start and index_end
    new_df = df
    for range in range_vector:
        mask = (new_df.index < range[0]) | (new_df.index > range[1])

        # Filter the DataFrame using the mask
        new_df = new_df[mask]

    # Reset the index to ensure continuous indices
    new_df.reset_index(drop=True, inplace=True)

    return new_df

def anomaly_detection(df, feature, contamination=0.05):
     # Initialize and fit the Isolation Forest model
    iso_forest = IsolationForest(contamination=contamination)  # Adjust the contamination parameter as needed
    iso_forest.fit(df[[feature]])

    # Predict anomalies (1 represents normal data, -1 represents anomalies)
    anomaly_predictions = iso_forest.predict(df[[feature]])

    # Add the anomaly predictions as a new column in the DataFrame
    df['anomaly'] = anomaly_predictions

    # Plot the data with highlighted anomalies
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df[feature], label='Data')
    plt.scatter(df.index[df['anomaly'] == -1], df[feature][df['anomaly'] == -1],
                color='red', label='Anomalies', marker='o')
    plt.xlabel('Index')
    plt.ylabel(feature)
    plt.title('Anomaly Detection using Isolation Forest')
    plt.legend()
    plt.show()

def read_datasets(folder_path):
    dataset_list = []

    files = os.listdir(folder_path)

    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            try:
                dataset = pd.read_csv(file_path)
                dataset['ID'] = file[:-4]
                dataset_list.append(dataset)
            except Exception as e:
                print(f"Error reading {file}: {e}")

    return dataset_list

def runmain_temp(df):
    # View a small portion
    range_vector = [[3000, 3050]]
    data_analysis_methods.plot_original(df, 'coppia', range_vector[0])

    #remove range that have unexpected behaviour (e.g., ramp-up, pauses)
    range_vector = [[0, 5000], [94000, 10000], [140000, 152063]]
    print('Index range lasts ', calculate_time_delta(df_948_H1, range_vector[1], 'minutes'))
    df = delete_rows_and_reset_index(df, range_vector)

    #data analysis methods
    data_analysis_methods.plot_original(df, 'temp', range_vector[1])
    df = data_analysis_methods.rolling_average_and_std(df, 'temp', 1000) # Create a plot
    data_analysis_methods.decomposition(df,'temp_rolling_mean', 10000)
    #data_analysis_methods.dickey_fuller_test(df_948_H1_rolling, 'temp_rolling_mean')
    #data_analysis_methods.autocorrelation_analysis(df_948_H1_rolling, 'temp')
    anomaly_detection(df, 'temp_rolling_mean')

    #!!!ARIMA takes 15-30 minutes with the full dataset ---> IT DOES NOT WORK WELL--> INVESTIGATION ONGOING
    range_vector=[[20000, 134999]]
    df = delete_rows_and_reset_index(df, range_vector)
    #data_analysis_methods.plot_original(df, 'temp', range_vector[0])
    #data_analysis_methods.arima(df_948_H1, 'temp_rolling_mean')

def runmain_coppia(df):
    # View a small portion
    range_vector = [[3000, 3050]]
    data_analysis_methods.plot_original(df, 'coppia', range_vector[0])

    #remove range that have unexpected behaviour (e.g., ramp-up, pauses)
    range_vector = [[0, 5000], [94000, 10000], [140000, 152063]]
    print('Index range lasts ', calculate_time_delta(df, range_vector[1], 'minutes'))
    #df_948_H1 = delete_rows_and_reset_index(df_948_H1, range_vector)

    #data analysis methods
    df = data_analysis_methods.rolling_average_and_std(df, 'coppia', 1000) # Create a plot
    data_analysis_methods.decomposition(df,'coppia_rolling_mean', 20000)
    #data_analysis_methods.dickey_fuller_test(df_948_H1_rolling, 'temp_rolling_mean')
    #data_analysis_methods.autocorrelation_analysis(df_948_H1_rolling, 'temp')
    anomaly_detection(df, 'coppia_rolling_mean')

    #ARIMA
    #!!!ARIMA takes 15-30 minutes with the full dataset ---> IT DOES NOT WORK WELL--> INVESTIGATION ONGOING
    range_vector=[[20000, 134999]]
    df = delete_rows_and_reset_index(df, range_vector)
    #data_analysis_methods.plot_original(df, 'coppia', range_vector[0])
    #data_analysis_methods.arima(df_948_H1, 'temp_rolling_mean')

def runmain():
    df_949_H2 = data_frame_init('/home/eliasmontini/PycharmProjects/AI_methods/datasets/test_msc/949_MSC1_H2.csv')
    df_948_H1 = data_frame_init('/home/eliasmontini/PycharmProjects/AI_methods/datasets/test_msc/948_MSC1_H1.csv')

    dataframes = read_datasets('/home/eliasmontini/PycharmProjects/AI_methods/datasets/test_msc')
    # Create a grid of plots
    num_datasets = len(dataframes)
    num_columns = num_datasets
    num_rows = 2  # 2 rows for the two decomposition results

    fig, axs = plt.subplots(num_rows, num_columns, figsize=(15, 8))

    for idx, dataset in enumerate(dataframes):
        ID = dataset['ID'].iloc[0]
        data_analysis_methods.rolling_average_and_std(dataset, 'coppia', 1000)  # Create a plot
        data_analysis_methods.rolling_average_and_std(dataset, 'temp', 1000)  # Create a plot

        decompose_temp = data_analysis_methods.decomposition(dataset, 'temp_rolling_mean', 10000, ID)
        decompose_coppia = data_analysis_methods.decomposition(dataset, 'coppia_rolling_mean', 10000, ID)

        # Extract the actual data from DecomposeResult objects
        decompose_temp_values = decompose_temp.resid
        decompose_coppia_values = decompose_coppia.resid

        axs[0, idx].plot(decompose_temp_values)  # Plot extracted data
        axs[0, idx].set_title(f'Dataset {idx + 1} Temp Decomposition')

        axs[1, idx].plot(decompose_coppia_values)  # Plot extracted data
        axs[1, idx].set_title(f'Dataset {idx + 1} Coppia Decomposition')

    plt.tight_layout()
    plt.show()

    print("df_950_H3")
    df_950_H3 = data_frame_init('/home/eliasmontini/PycharmProjects/AI_methods/datasets/test_msc/957_MSC1_H3.csv')
    data_analysis_methods.rolling_average_and_std(df_950_H3, 'temp', 1000)  # Create a plot
    decompose_temp = data_analysis_methods.decomposition(df_950_H3, 'temp_rolling_mean', 10000, '957_MSC1_H3')
    range_vector = [[10000, 150000]]
    df_950_H3 = delete_rows_and_reset_index(df_950_H3, range_vector)
    decompose_temp = data_analysis_methods.decomposition(df_950_H3, 'temp_rolling_mean', 1000, '957_MSC1_H3')


if __name__ == "__main__":
    runmain()
