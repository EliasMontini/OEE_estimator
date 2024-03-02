import pandas as pd
import data_initialisation
import data_analysis_methods
import matplotlib.pyplot as plt

OS = "windows"
if OS == "ubuntu":
    dataset = data_initialisation.data_frame_init(
    '/home/eliasmontini/PycharmProjects/AI_methods/datasets/test_msc/957_MSC1_H3.csv')
else:
    dataset = data_initialisation.data_frame_init(
    'C:\\Users\\emontini.inst\\PycharmProjects\\data_analysis_INT\\datasets\\test_msc\\956_MSC1_H1.csv')



data_analysis_methods.rolling_average_and_std(dataset, 'coppia', 1000)  # Create a plot

data_analysis_methods.rolling_average_and_std(dataset, 'temp', 1000)  # Create a plot
data_analysis_methods.plot_original(dataset, 'temp_rolling_mean', [0, 150000])

decompose_temp = data_analysis_methods.decomposition(dataset, 'temp_rolling_mean', 1000, "Temp")
decompose_coppia = data_analysis_methods.decomposition(dataset, 'coppia_rolling_mean', 1000, "Coppia")




dataset_cutted = data_initialisation.data_frame_init(
    '/home/eliasmontini/PycharmProjects/AI_methods/datasets/test_msc/957_MSC1_H3.csv')
dataset_cutted = data_initialisation.delete_rows_and_reset_index(dataset, [[50000, 151968]])


data_analysis_methods.rolling_average_and_std(dataset_cutted, 'coppia', 1000)  # Create a plot
data_analysis_methods.rolling_average_and_std(dataset_cutted, 'temp', 1000)  # Create a plot

decompose_temp_cutted = data_analysis_methods.decomposition(dataset_cutted, 'temp_rolling_mean', 10000, "Temp")
decompose_coppia_cutted = data_analysis_methods.decomposition(dataset_cutted, 'coppia_rolling_mean', 10000, "Coppia")




