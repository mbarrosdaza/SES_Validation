# -*- coding: utf-8 -*-
"""
Created on Thu May  1 19:13:00 2025

@author: mdaza
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.colors as pltc
import scipy.spatial as scsp
import mpl_toolkits.mplot3d as a3
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


##################### MEASURED TEMPERATURE VALUES - EXPERIMENTS #########################################################################

loops = [202, 301, 302, 303, 304, 205, 305,
         306, 307, 207, 208, 209, 211, 213, 214]
loops_length_ft = [2736, 2373, 2236, 2116, 2059, 2019,
                   1982, 1923, 1816, 1668, 1399, 1053, 692, 347, 65]
loops_length_m = [ft * 0.3048 for ft in loops_length_ft]

cold_measured_data = pd.read_csv('./measured_data/cold_flow_tests.csv')

#### 83% Cold Flow Analysis #################################################################################################################

ses_result_files = glob.glob('./SES_results/*.xlsx')
cold = []
cold_2 = []
for file in ses_result_files:
    if file.endswith('JFs.xlsx'):
        data = pd.read_excel(file, sheet_name="flow_rate")
        cold.append(data.iloc[:, 65:106].mean(axis=1).mean(axis=0))
    if file.endswith('JFs_075.xlsx'):
        data = pd.read_excel(file, sheet_name="Flow Rate")
        cold_2.append(data.iloc[:, 65:106].mean(axis=1).mean(axis=0))
cold.sort()
cold_2.sort()
jetfansnumber = [1, 2, 3, 4,5, 6, 7, 8, 9,10, 11,12,13, 14, 15]
SES_results = pd.DataFrame({"Fans Number": jetfansnumber, "flow_rate": cold})

SES_results_2 = pd.DataFrame({"Fans Number": jetfansnumber, "flow_rate": cold_2})

plt.plot(cold_measured_data["Fans Number"], cold_measured_data["flow_rate_exp_m3_s"],
         label="Measured", marker="o", color='black', fillstyle='none')
plt.scatter(SES_results_2["Fans Number"], SES_results_2["flow_rate"],
            label="SES-75%", marker="s", color='blue')
plt.scatter(SES_results["Fans Number"], SES_results["flow_rate"],
            label="SES-83%", marker="v", color='green')
plt.yticks([0, 100, 200, 300, 400])
plt.xticks([1, 3, 5, 7, 9, 11, 13, 15])
plt.legend(fontsize=10)
plt.xlabel('Number of jet fans', fontsize=16)
plt.ylabel('Flowrate (m³/s)', fontsize=16)
plt.tight_layout()

plt.savefig('cold_flow_plot.png', dpi=300)

plt.show()


#### 83% Cold Flow Analysis #################################################################################################################



###################### COMPARISON PLOT ##############################################################

measured = cold_measured_data["flow_rate_exp_m3_s"].to_numpy()
predicted = SES_results["flow_rate"].to_numpy()

######################################################################
error = predicted - measured
error_percentage = np.abs(error) / measured * 100
avg_error = np.mean(error_percentage)
mean = np.mean(error)
std = np.std(error)
r2 = r2_score(measured, predicted)
#################################################################
predicted_2 = SES_results_2["flow_rate"].to_numpy()

error_2 = predicted_2 - measured
error_percentage_2 = np.abs(error_2) / measured * 100
avg_error_2 = np.mean(error_percentage_2)
mean_2 = np.mean(error_2)
std_2 = np.std(error_2)
r2_2 = r2_score(measured, predicted_2)


plt.figure()
plt.scatter(measured, predicted_2,
            marker="s", color='blue',  label='SES-75%')
plt.scatter(measured, predicted,
            marker="v", color='green', label='SES-83%')

lims = [min(np.min(measured), np.min(predicted)), max(np.max(measured), np.max(predicted))]
plt.plot(lims, lims, '--', color='grey', label='1:1 Line')
plt.xlim(lims)
plt.ylim(lims)


plt.gca().text(0.92, 0.27, 'SES-75%', fontsize=8, transform=plt.gca().transAxes, verticalalignment='bottom',
               horizontalalignment='right')
plt.gca().text(0.95, 0.05,
               f'Avg Error: {avg_error_2:.1f}%\nMean Error: {mean_2:.2f} m³/s \nStd Dev: {std_2:.2f} m³/s \nR²: {r2_2:.3f}',
               fontsize=8, transform=plt.gca().transAxes,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(facecolor='white', alpha=0.7))

plt.gca().text(0.675, 0.27, 'SES-83%', fontsize=8, transform=plt.gca().transAxes, verticalalignment='bottom',
               horizontalalignment='right')
plt.gca().text(0.675, 0.05,
               f'Avg Error: {avg_error:.1f}%\nMean Error: {mean:.2f} m³/s \nStd Dev: {std:.2f} m³/s \nR²: {r2:.3f}',
               fontsize=8, transform=plt.gca().transAxes,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(facecolor='white', alpha=0.7))

plt.xlabel('Measured Cold Flowrate (m³/s)', fontsize=16)
plt.ylabel('Predicted Cold Flowrate (m³/s)', fontsize=16)
plt.title('Cold Flowrate Prediction Comparison', fontsize=12)
plt.legend(fontsize=10)
plt.tight_layout()

plt.savefig('cold_flow_stat.png', dpi=300)

plt.show()
