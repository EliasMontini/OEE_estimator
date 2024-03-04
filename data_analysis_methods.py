import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima
from math import sqrt
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
from plotly.subplots import make_subplots
import statsmodels.api as sm
import plotly.graph_objects as go
from sklearn.impute import SimpleImputer
import itertools
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import pmdarima as pm
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def interactive_plot(values, title):
    # Assuming decomposition_result contains your decomposition data
    # You might need to adjust the following lines based on your actual data structure

    # Create a figure using plotly express
    fig = px.line(values)
    fig.update_layout(title=title)

    # Show the interactive plot
    fig.show()


def add_timedelta(df, timestamp):
    df['time_diff'] = df[timestamp].transform(lambda x: x - x.min())
    df['time_diff'] = pd.to_timedelta(df['time_diff'])  # convert the column into a datetime object
    print(df.head())


def plot_original(df, feature, range):
    # Create a plot
    plt.figure(figsize=(10, 6))
    # Select data within the specified index range
    subset_df = df.loc[range[0]:range[1]]
    interactive_plot(subset_df[feature], feature)


def rolling_average_and_std(df, feature, rolling_window):
    # Calculate rolling mean
    df[feature + '_rolling_mean'] = df[feature].rolling(window=rolling_window, min_periods=1).mean().reset_index(
        drop=True)
    df[feature + 'rolling_std'] = df[feature].rolling(window=rolling_window).std()
    return df


def dickey_fuller_test(df,
                       feature):  # We can see if our data is not stationary from the p-value. >5--> not stationary; <=5 stationary
    # Handle missing or NaN values
    df[feature].fillna(method='ffill', inplace=True)  # Forward fill missing values

    # Perform ADF test
    print("Waiting for dickey_fuller_test")
    result = adfuller(df[feature])
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[4])

    # Interpret the results
    if result[1] <= 0.05:
        print("Reject the null hypothesis: The data is stationary.")
    else:
        print("Fail to reject the null hypothesis: The data is not stationary.")


def autocorrelation_analysis(df, feature):
    # Calculate autocorrelation with different lags
    lags = [5 * 12, 5 * 12 * 60, 5 * 12 * 60 * 24,
            5 * 12 * 60 * 24 * 7]  # Lag in observations (minute, hour, day, week)
    autocorr_values = []

    for lag in lags:
        autocorr = df[feature].autocorr(lag=lag)
        autocorr_values.append(autocorr)
        lag_duration = lag * 5  # Convert lag back to seconds
        print(f"Autocorrelation with lag {lag_duration} seconds: {autocorr:.2f}")

    # Plot autocorrelation values
    plt.bar(lags, autocorr_values, tick_label=[f"{lag_duration} seconds" for lag_duration in lags])
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation at Different Lags')
    plt.show()


def decomposition(df, feature, period):
    dickey_fuller_test(df, feature) # We can see if our data is not stationary from the p-value. >5--> not stationary; <=5 stationary
    # Handle missing or NaN values

    df[feature].fillna(method='ffill', inplace=True)  # Forward fill missing values
    decompose = seasonal_decompose(df[feature], period=period, model='additive')

    # Extract components
    decompose_df = pd.DataFrame({
        'Date': df.index,
        'Observed': decompose.observed,
        'Trend': decompose.trend,
        'Seasonal': decompose.seasonal,
        'Residual': decompose.resid
    })

    # Create a subplot with 4 rows and 1 column
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        vertical_spacing=0.1, subplot_titles=('Observed', 'Trend', 'Seasonal', 'Residual'))

    # Add each component to the corresponding subplot
    for i, component in enumerate(['Observed', 'Trend', 'Seasonal', 'Residual']):
        fig.add_trace(go.Scatter(x=decompose_df.index, y=decompose_df[component], mode='lines', name=component),
                      row=i + 1, col=1)

        # Update y-axis label for each subplot
        fig.update_yaxes(title_text='Value', row=i + 1, col=1)

    # Update the layout
    fig.update_layout(title=f'Seasonal Decomposition of {feature}', template='plotly')

    fig.show()

    # Update subplot titles and layout
    fig.update_layout(title='Seasonal Decomposition',
                      xaxis=dict(title='Date'),
                      showlegend=True,
                      height=800)

    # Show the Plotly figure
    fig.show()
    return decompose_df


def arima(df, feature_to_be_estimated):
    # Calculate the split point as a percentage of the total length
    split_percentage = 0.95  # 80% training, 20% testing
    split_index = int(len(df.index) * split_percentage)

    # Split the dataframe into training and testing sets
    train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()
    train_values = train[feature_to_be_estimated].values
    test_values = test[feature_to_be_estimated].values
    model = auto_arima(train_values, trace=True, error_action='ignore', suppress_warnings=True)
    model.fit(train_values)
    forecast = model.predict(n_periods=len(test))
    forecast = pd.DataFrame(forecast, index=test.index, columns=['Prediction'])
    rms = sqrt(mean_squared_error(test_values, forecast))
    print("RMSE: ", rms)

    # Plotting
    fig = px.line()
    fig.add_scatter(x=train.index, y=train[feature_to_be_estimated], mode='lines', name='Train')
    fig.add_scatter(x=test.index, y=test[feature_to_be_estimated], mode='lines', line=dict(color='red'), name='Test')
    fig.add_scatter(x=forecast.index, y=forecast['Prediction'], mode='lines', line=dict(color='black'), name='Forecast')

    fig.update_layout(
        title="Train/Test split",
        xaxis_title="Index",
        yaxis_title=feature_to_be_estimated,
        legend=dict(x=0, y=1, traceorder='normal', orientation='h'),
    )

    fig.show()


def anomaly_detection(df, feature_of_interest):
    df.fillna(method='ffill', inplace=True)  # Forward fill missing values
    df_values = df[feature_of_interest].values.reshape(-1, 1)

    # Handle NaN values using SimpleImputer (mean imputation in this example)
    imputer = SimpleImputer(strategy='mean')
    df_values_imputed = imputer.fit_transform(df_values)

    # Perform anomaly detection using Isolation Forest on the imputed data
    outliers_fraction = 0.01
    model = IsolationForest(contamination=outliers_fraction, random_state=42)
    model.fit(df_values_imputed)

    anomaly_labels = model.predict(df_values_imputed)

    # Create a new column in the DataFrame to indicate anomalies
    df['Anomaly'] = anomaly_labels

    # Create a trace for the real dataset (line plot)
    trace_real_data = go.Scatter(x=df.index, y=df[feature_of_interest], mode='lines', name='Real Data')

    # Create a trace for the anomalies (scatter plot)
    trace_anomalies = go.Scatter(x=df[df['Anomaly'] == -1].index,
                                 y=df[df['Anomaly'] == -1][feature_of_interest],
                                 mode='markers',
                                 name='Anomalies',
                                 marker=dict(color='red', size=8))

    # Plot the real dataset and anomalies using Plotly
    fig = go.Figure([trace_real_data, trace_anomalies])
    fig.update_layout(title='Anomaly Detection using Isolation Forest',
                      xaxis=dict(title='Time'),
                      yaxis=dict(title='Value'))

    return fig.show()



def anomaly_detection_with_comparison(df_to_compare, reference_dataset_vector, feature_of_interest_vector):
    max_size = len(df_to_compare)
    similarity_scores = []

    # Choose the maximum size of the smallest dataset
    for ref_dataset in reference_dataset_vector:
        if len(ref_dataset) < max_size:
            max_size = len(ref_dataset)
        ref_dataset.fillna(method='ffill', inplace=True)  # Forward fill missing values

    for ref_dataset in reference_dataset_vector:
        df_to_compare = df_to_compare[feature_of_interest_vector][:max_size]
        distances = np.linalg.norm(df_to_compare - ref_dataset, axis=1)
        similarity_scores.append(distances.mean())

    df_to_compare.fillna(method='ffill', inplace=True)  # Forward fill missing values

    # Print similarity scores for each reference dataset
    for i, score in enumerate(similarity_scores):
        print(f"Similarity score with dataset {i + 1}: {score}")

    # Perform anomaly detection using Isolation Forest
    outliers_fraction = 0.05  # You can adjust this based on your dataset characteristics
    model = IsolationForest(contamination=outliers_fraction, random_state=42)
    print(df_to_compare)
    reference_dataset_vector = pd.concat(reference_dataset_vector, axis=0)[feature_of_interest_vector]
    model.fit(reference_dataset_vector)

    anomaly_labels = model.predict(df_to_compare)

    # Print anomalies detected
    anomalies = df_to_compare[anomaly_labels == -1]
    print("Detected anomalies:")
    print(anomalies)

    # Create a plot to visualize the anomalies
    fig = px.line(title='Time Series with Anomalies')

    # Plot the time series data
    for feature_of_interest in feature_of_interest_vector:
        fig.add_scatter(x=df_to_compare.index, y=df_to_compare[feature_of_interest], mode='lines',
                        name=feature_of_interest)

        # Highlight anomalies with red circles
        anomaly_indices = np.where(anomaly_labels == -1)[0]
        fig.add_scatter(x=df_to_compare.index[anomaly_indices],
                        y=df_to_compare.iloc[anomaly_indices][feature_of_interest],
                        mode='markers', marker=dict(color='yellow'), name='Anomaly ' + feature_of_interest)

    fig.update_xaxes(title='Time')
    fig.update_yaxes(title='Values')
    fig.show()


def test_anomaly():
    columns_of_interest = ['temp_rolling_mean']
    dataset1 = pd.read_csv(
        '/home/eliasmontini/PycharmProjects/AI_methods/datasets/test_msc/949_MSC1_H2.csv')  # replace with your actual file paths
    dataset2 = pd.read_csv('/home/eliasmontini/PycharmProjects/AI_methods/datasets/test_msc/935_MSC1_H2.csv')
    dataset3 = pd.read_csv('/home/eliasmontini/PycharmProjects/AI_methods/datasets/test_msc/936_MSC1_H3.csv')
    dataset_to_compare = pd.read_csv('/home/eliasmontini/PycharmProjects/AI_methods/datasets/test_msc/957_MSC1_H3.csv')
    rolling_window = 10000
    dataset1 = rolling_average_and_std(dataset1, 'temp', rolling_window)
    dataset1 = rolling_average_and_std(dataset1, 'coppia', rolling_window)
    dataset2 = rolling_average_and_std(dataset2, 'temp', rolling_window)
    dataset2 = rolling_average_and_std(dataset2, 'coppia', rolling_window)
    dataset3 = rolling_average_and_std(dataset3, 'temp', rolling_window)
    dataset3 = rolling_average_and_std(dataset3, 'coppia', rolling_window)
    dataset_to_compare = rolling_average_and_std(dataset_to_compare, 'temp', rolling_window)
    dataset_to_compare = rolling_average_and_std(dataset_to_compare, 'coppia', rolling_window)
    anomaly_detection_with_comparison(dataset_to_compare, [dataset1, dataset2, dataset3], columns_of_interest)


def sarimax(df, feature_to_be_estimated):
    data_stationarized = df[feature_to_be_estimated].diff()[1:]
    sm.graphics.tsa.plot_acf(data_stationarized)
    plt.show()
    sm.graphics.tsa.plot_pacf(data_stationarized)
    plt.show()

    # Calculate the split point as a percentage of the total length
    split_percentage = 0.99  # 80% training, 20% testing
    split_index = int(len(df.index) * split_percentage)

    # Split the dataframe into training and testing sets
    train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()
    train_values = train[feature_to_be_estimated].values
    test_values = test[feature_to_be_estimated].values

    # Define the parameter grid to iterate over
    p_values = [0, 1, 2]
    d_values = [0, 1]
    q_values = [0, 1, 2]
    seasonal_order = [(1, 0, 0, 12), (0, 1, 1, 12),(2, 0, 0, 12)]

    best_rmse = float('inf')
    best_params = None
#-------------------------------------------
    print('Training is starting ...')

    #model = sm.tsa.SARIMAX(endog=train_values, order=(2, 0, 0), seasonal_order=(2, 0, 0, 12))
    model = sm.tsa.SARIMAX(endog=train_values, order=(4, 1, 0), seasonal_order=(0, 1, 2, 12))

    model_fit = model.fit()
    # Forecast for the entire length of the test set
    forecast = model_fit.forecast(steps=len(test_values))

    # Calculate RMSE (root mean squared error)
    rmse = mean_squared_error(y_true=test_values, y_pred=forecast, squared=False)
    print("Root Mean Squared Error:", rmse)
#--------------------------------------
  #  Iterate over parameters
  #   for p, d, q, seasonal in itertools.product(p_values, d_values, q_values, seasonal_order):
  #       model = SARIMAX(train_values, order=(p, d, q), seasonal_order=seasonal)
  #       model_fit = model.fit()
  #       forecast = model_fit.forecast(steps=len(test_values))
  #       rmse = mean_squared_error(test_values, forecast,squared=False)
  #
  #       print("Order", p, d, q," Seasonal", seasonal,"--> RMSE = ", rmse)
  #       if rmse < best_rmse:
  #           best_rmse = rmse
  #           best_params = (p, d, q, seasonal)
  #
  #   # Fit the best SARIMAX model
  #   best_model = SARIMAX(endog=train_values, order=(best_params[0], best_params[1], best_params[2]),
  #                        seasonal_order=best_params[3])
  #   best_model = best_model.fit()
  #   forecast = best_model.forecast(steps=len(test_values))
# --------------------------------------
    forecast_df = pd.DataFrame(data=forecast, columns=['Prediction'], index=test.index)

    # Plotting
    fig = px.line()
    fig.add_scatter(x=train.index, y=train[feature_to_be_estimated], mode='lines', name='Train')
    fig.add_scatter(x=test.index, y=test[feature_to_be_estimated], mode='lines', line=dict(color='red'), name='Test')
    fig.add_scatter(x=forecast_df.index, y=forecast_df['Prediction'], mode='lines', line=dict(color='black'),
                    name='Forecast')
    fig.update_layout(
        title="Train/Test split",
        xaxis_title="Index",
        yaxis_title=feature_to_be_estimated,
        legend=dict(x=0, y=1, traceorder='normal', orientation='h'),
    )

    fig.show()


def auto_sarimax(df, feature_to_be_estimated):

    # Calculate the split point as a percentage of the total length
    split_percentage = 0.8
    split_index = int(len(df.index) * split_percentage)

    # Split the dataframe into training and testing sets
    train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()

    train_values = train[feature_to_be_estimated].values
    test_values = test[feature_to_be_estimated].values
    best_aic = float("inf")
    best_m = None
    best_model = None

    # Auto-SARIMA model fitting
    # Loop through potential m values (e.g., from 1 to 24)
    print("Training is starting...")
    for m in range(12, 12):
        # Assuming the range of possible m values is from 1 to 24
        print("m --> ", m)
        temp_model = pm.auto_arima(train_values, seasonal=True, m=m,
                                   stepwise=True, trace=False,
                                   error_action='ignore', suppress_warnings=True)

        if temp_model.aic() < best_aic:
            best_aic = temp_model.aic()
            best_m = m
            best_model = temp_model

    print(f"Best AIC is {best_aic} for seasonality period m = {best_m}")
    # Print best model summary
    print(best_model.summary())

    # Forecast
    forecast = best_model.predict(n_periods=len(test))

    # Calculate RMSE
    step_size = 100
    rmse = mean_squared_error(test_values, forecast, squared=False)
    print(f'RMSE: {rmse}')

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train_values, label='Train')
    plt.plot(test.index, test_values, label='Test')
    plt.plot(test.index, forecast, label='Forecast')
    plt.legend(loc='upper left', fontsize=12)
    plt.show()

    # Plotting
    fig = px.line()
    fig.add_scatter(x=train.index, y=train[feature_to_be_estimated], mode='lines', name='Train')
    fig.add_scatter(x=test.index, y=test[feature_to_be_estimated], mode='lines', line=dict(color='red'), name='Test')
    fig.add_scatter(x=forecast, y=forecast, mode='lines', line=dict(color='black'),
                    name='Forecast')
    fig.update_layout(
        title="Train/Test split",
        xaxis_title="Index",
        yaxis_title=feature_to_be_estimated,
        legend=dict(x=0, y=1, traceorder='normal', orientation='h'),
    )

    fig.show()


    # Initialize RMSE calculation
    total_rmse = 0
    count = 0

    # Rolling forecast
    step_size = 100
    # Rolling forecast
    forecasted_values = []
    test_values_list = []
    for start in range(0, len(test) - step_size + 1, step_size):
        test = test.iloc[start:start + step_size]
        test_values = test[feature_to_be_estimated].values

        forecast = best_model.predict(n_periods=len(test))

        rmse = mean_squared_error(test_values, forecast, squared=False)
        total_rmse += rmse
        count += 1

        # Append the test and forecast values for plotting
        test_values_list.extend(test_values)
        forecasted_values.extend(forecast)

        # Update the model with the new test data
        best_model.update(test_values)

    # Calculate average RMSE
    avg_rmse = total_rmse / count
    print(f'Average RMSE over all rolling steps: {avg_rmse}')

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[:split_index], train[feature_to_be_estimated], label='Train')
    plt.plot(df.index[split_index:split_index + len(test_values_list)], test_values_list, label='Test')
    plt.plot(df.index[split_index:split_index + len(forecasted_values)], forecasted_values,
             label='Forecast')
    plt.legend(loc='upper left', fontsize=12)
    plt.show()

