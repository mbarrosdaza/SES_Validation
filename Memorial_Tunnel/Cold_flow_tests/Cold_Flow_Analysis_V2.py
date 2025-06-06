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
from sklearn.metrics import mean_absolute_error, mean_squared_error

##################### MEASURED TEMPERATURE VALUES - EXPERIMENTS #########################################################################

loops = [202, 301, 302, 303, 304, 205, 305,
         306, 307, 207, 208, 209, 211, 213, 214]
loops_length_ft = [2736, 2373, 2236, 2116, 2059, 2019,
                   1982, 1923, 1816, 1668, 1399, 1053, 692, 347, 65]
loops_length_m = [ft * 0.3048 for ft in loops_length_ft]

cold_measured_data = pd.read_csv('./measured_data/cold_flow_tests.csv')

#### Cold Flow Analysis #################################################################################################################

ses_result_files = glob.glob('./SES_results/*.xlsx')
cold = []
for file in ses_result_files:
    data = pd.read_excel(file, sheet_name="flow_rate")
    cold.append(data.iloc[:, 65:106].mean(axis=1).mean(axis=0))
cold.sort()

jetfansnumber = [1, 2, 3, 4,5, 6, 7, 8, 9,10, 11,12,13, 14, 15]
SES_results = pd.DataFrame({"Fans Number": jetfansnumber, "flow_rate": cold})

plt.plot(cold_measured_data["Fans Number"], cold_measured_data["flow_rate_exp_m3_s"],
         label="Measured", marker="o", color='black', fillstyle='none')
plt.scatter(SES_results["Fans Number"], SES_results["flow_rate"],
            label="SES", marker="s", color='blue')
plt.yticks([0, 100, 200, 300, 400])
plt.xticks([1, 3, 5, 7, 9, 11, 13, 15])
plt.legend(fontsize=10)
plt.xlabel('Number of jet fans', fontsize=16)
plt.ylabel('Flowrate (m³/s)', fontsize=16)
plt.tight_layout()
plt.show()

###################### COMPARISON PLOT ##############################################################

measured = cold_measured_data["flow_rate_exp_m3_s"].to_numpy()
predicted = SES_results["flow_rate"].to_numpy()

error = predicted - measured
error_percentage = np.abs(error) / measured * 100
avg_error = np.mean(error_percentage)
mean = np.mean(error)
std = np.std(error)

plt.figure()
plt.scatter(measured, predicted,
            marker='o', color='black', facecolors='none', label='Cold Flow Points')

lims = [min(np.min(measured), np.min(predicted)), max(np.max(measured), np.max(predicted))]
plt.plot(lims, lims, '--', color='grey', label='1:1 Line')
plt.xlim(lims)
plt.ylim(lims)

plt.gca().text(0.95, 0.05,
               f'Avg Error: {avg_error:.1f}%\nMean Error: {mean:.2f} m³/s \nStd Dev: {std:.2f} m³/s',
               fontsize=10, transform=plt.gca().transAxes,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(facecolor='white', alpha=0.7))

plt.xlabel('Measured Cold Flowrate (m³/s)', fontsize=16)
plt.ylabel('Predicted Cold Flowrate (m³/s)', fontsize=16)
plt.title('Cold Flowrate Prediction Comparison', fontsize=12)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()
