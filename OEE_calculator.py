from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np

# Function to calculate OEE based on the chosen method
def calculate_oee(data, method, variables, thresholds, threshold_conditions_on, rolling_period=None, model_path=None, device_daily_hours=None):
    data_copy = data.copy()

    if method == 1:
        # Method 1: Threshold-based classification for multiple variables with different assessment criteria
        for i, variable in enumerate(variables):
            threshold = thresholds[i]
            assessment = threshold_conditions_on[i]
            if assessment == "above":
                data_copy[f'classified_{variable}'] = data[variable] > threshold
            elif assessment == "less":
                data_copy[f'classified_{variable}'] = data[variable] < threshold
            elif assessment == "equal":
                data_copy[f'classified_{variable}'] = data[variable] == threshold

        # Assuming "on" if any of the variables meets its threshold assessment criteria
        data_copy['classified'] = data_copy[[f'classified_{var}' for var in variables]].all(axis=1).map(
            {True: 'on', False: 'off'})

    elif method == 2:
        # Method 2: Rolling average method for multiple variables with different assessment criteria
        for i, variable in enumerate(variables):
            rolling_threshold = thresholds[i]
            assessment = threshold_conditions_on[i]
            data_copy[f'{variable}_rolling_avg'] = data_copy[variable].rolling(window=rolling_period).mean()
            if assessment == "above":
                data_copy[f'classified_{variable}'] = data_copy[f'{variable}_rolling_avg'] > rolling_threshold
            elif assessment == "less":
                data_copy[f'classified_{variable}'] = data_copy[f'{variable}_rolling_avg'] < rolling_threshold
            elif assessment == "equal":
                data_copy[f'classified_{variable}'] = data_copy[f'{variable}_rolling_avg'] == rolling_threshold

        # Assuming "on" if any of the variables' rolling averages meets its threshold assessment criteria
        data_copy['classified'] = data_copy[[f'classified_{var}' for var in variables]].all(axis=1).map(
            {True: 'on', False: 'off'})

    elif method == 3:
        # Delete rows with missing values for specified variables
        data_copy = data_copy.dropna(subset=variables)
        # Load the saved model
        model = joblib.load(model_path)

        X = data_copy[variables]
        # Use the model to make predictions
        predictions = model.predict(X)
        # Assuming the model's predictions are binary (1 for 'on', 0 for 'off')
        # You can adjust this logic based on how your model's predictions are structured
        data_copy['classified'] = predictions
        data_copy['classified'] = data_copy['classified'].map({1: 'on', 0: 'off'})
    else:
        raise ValueError("Invalid OEE estimation method.")

    # plot
    temp_variable = variables[0]
    # Segment the data by 'classified' changes
    unique_classified = data_copy['classified'].unique()
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_classified)))  # Generate colors
    classified_dict = dict(zip(unique_classified, colors))

    fig, ax = plt.subplots()
    for classified, group_df in data_copy.groupby('classified'):
        # Using scatter plot here
        ax.scatter(group_df['timestamp'], group_df[temp_variable], label=classified,
                   color=classified_dict[classified])

    plt.legend(title='Classified')
    plt.title('Temperature Over Time by Classification')
    plt.xlabel('Timestamp')
    plt.ylabel(temp_variable.capitalize())
    plt.show()

    # Calculate OEE
    print("min ", data_copy['timestamp'].min())
    print("max ", data_copy['timestamp'].max())
    print(data_copy)
    if device_daily_hours == None:
        total_time = (data_copy['timestamp'].max() - data_copy['timestamp'].min()).total_seconds()
    else:
        total_time = device_daily_hours * (data_copy['timestamp'].max() - data_copy['timestamp'].min()).days
    print("Total time ", total_time, " hours")
    productive_time = data_copy[data_copy['classified'] == 'on']['timestamp'].count() * (total_time / len(data_copy))
    print("Productive time ", productive_time, " hours")

    oee = (productive_time / total_time) *100


    return oee

def calculate_oee_method_3(data, variables, file_path):
        # Delete rows with missing values for specified variables
        data_copy = data.copy()
        data_copy = data_copy.dropna(subset=variables)
        if data_copy['state'].isnull().any():
            print("Warning: Missing values found in 'state' column.")
            data_copy = data_copy.dropna(subset="state")

        # Split the data into features (X) and target (y)
        X = data_copy[variables]
        y = data_copy['state']

        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Initialize and train a more complex model, e.g., RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict and calculate accuracy
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model accuracy: {accuracy}")

        # Save the model to the same folder as the data
        model_filename = 'oee_model.pkl'
        data_file_path =   file_path # Adjust the path according to your environment
        model_path = os.path.join(os.path.dirname(data_file_path), model_filename)
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")

def thresholds_definition(threshold_method, product_key=None, thresholds_generic = None, thresholds_specific=None):
    """
    Define thresholds based on the specified method.

    Parameters:
    - threshold_method: A string specifying the thresholding method ('static' or 'dynamic_with_average').
    - product_key: A string specifying the product for specific thresholds.
    - thresholds_static: A list specifying static thresholds for variables (applies if generic static thresholds are used).
    - specific_thresholds: A dictionary mapping product keys to their specific static thresholds.

    Returns:
    - thresholds: The determined thresholds based on the method and parameters provided.
    """

    thresholds = None

    if threshold_method == 'static':
        # Check if specific thresholds for a product are requested
        if product_key and thresholds_specific:
            # Attempt to retrieve specific thresholds for the given product
            thresholds = thresholds_specific.get(product_key)
            if thresholds is None:
                print(
                    f"No specific thresholds found for product {product_key}. Falling back to generic static thresholds.")
                thresholds = thresholds_generic
        else:
            # Use generic static thresholds
            thresholds = thresholds_generic

    elif threshold_method == 'dynamic_with_average':
        # Placeholder for dynamic threshold calculation logic
        print("Dynamic with anomaly detection...")

    return thresholds


def filter_out_periods(data, date_ranges):
    # Convert date_ranges to datetime
    date_ranges_dt = [(pd.to_datetime(start, dayfirst=True), pd.to_datetime(end, dayfirst=True)) for start, end in
                      date_ranges]

    # Initialize a mask to track rows to keep
    mask = pd.Series([False] * len(data), index=data.index)

    # Apply each date range as a filter, updating the mask
    for start, end in date_ranges_dt:
        mask |= ((data['timestamp'] >= start) & (data['timestamp'] <= end))

    # Find rows that fall outside all specified ranges
    outside_ranges = data[~mask]

    if not outside_ranges.empty:
        print("There are rows outside the specified date ranges.")
        # Optionally, print out some of these rows or their count
        print(f"Count of rows outside specified ranges: {len(outside_ranges)}")
    else:
        print("All rows fall within the specified date ranges.")

    # Filter out rows that fall within any of the specified ranges
    data_filtered = data[~mask]

    return data_filtered


def main():
    # Relative path or parameter for dataset
    file_path = './data/OEEtest_Modified_ms_with_state.csv'

    data = pd.read_csv(file_path)

    # Preprocess the timestamp to a readable format if necessary
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    device_daily_hours = 8
    # Example of placeholders and configurations
    variables = ["temp"] # [ "coppia", "temp",]
    thresholds_device_based = [60] #[0.1, 40]
    threshold_conditions_on = ["above"] #["above", "above"]

    date_ranges = [
        ["2023-08-21 12:13:20", "2023-08-21 12:13:21"],
        ["01/03/2022 00:00:00", "01/04/2022 23:59:59"],
    ]
    rolling_period = 5
    OEE_estimation_method = 3
    model_path = './data/oee_model.pkl'


    # Check if all lists have the same length
    if len(variables) == len(thresholds_device_based) == len(threshold_conditions_on):
        print("All lists have the same length.")
    else:
        raise ValueError("Lists do not have the same length.")

    # Example function calls with placeholders
    thresholds = thresholds_device_based  # Simplified for demonstration
    calculate_oee_method_3(data, variables, file_path)
    generic_thresholds = thresholds  # Simplified for demonstration
    print(generic_thresholds)

    data = filter_out_periods(data, date_ranges)
    OEE = calculate_oee(data, OEE_estimation_method, variables, generic_thresholds, threshold_conditions_on, rolling_period,
                        model_path, device_daily_hours)

    print("OEE ", OEE, "%")


if __name__ == "__main__":
    main()
