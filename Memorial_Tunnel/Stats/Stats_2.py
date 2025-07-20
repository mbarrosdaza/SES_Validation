# -*- coding: utf-8 -*-
"""
Created on Tue May  6 08:37:58 2025

@author: mdaza
"""

import pandas as pd
import matplotlib.pyplot as plt
import glob
import itertools
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


markers = ['s', 'o', '^', 'D', 'P']  # square, circle, triangle, diamond, plus-filled

##################### PREDICTED SES VALUES ##################################################################################
############################################################################################################################
############################################################################################################################

SES_results_files = ['../Test_606A/Outputs/SES_results_606A.xlsx', '../Test_607/Outputs/SES_results_607.xlsx',  '../Test_611/Outputs/SES_results_611.xlsx',           
                     '../Test_615B/Outputs/SES_results_615B.xlsx']
      
Q_UPST_SES = []
Q_DOWN_SES = []
T_UPST_SES = []
T_DOWN_SES = []

for file in SES_results_files:
    xls = pd.ExcelFile(file)
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        Q_UPST = df.loc[df['length'] < 605, 'flowrate'].mean()
        Q_DOWN = df.loc[df['length'] > 605, 'flowrate'].mean()
        T_UPST = df.loc[df['length'] < 605, 'temperature'].mean()
        T_DOWN = df.loc[df['length'] > 605, 'temperature'].mean()
        Q_UPST_SES.append(Q_UPST)
        Q_DOWN_SES.append(Q_DOWN)
        T_UPST_SES.append(T_UPST)
        T_DOWN_SES.append(T_DOWN)
        
##################### MEASURED VALUES ######################################################################################
############################################################################################################################
############################################################################################################################
    
files_flow = ['../Test_606A/Outputs/T606A_Q.csv', '../Test_607/Outputs/T607_Q.csv',
              '../Test_611/Outputs/T611_Q.csv', '../Test_615B/Outputs/T615B_Q.csv']

files_temp = ['../Test_606A/Outputs/T606A_T.csv', '../Test_607/Outputs/T607_T.csv',
              '../Test_611/Outputs/T611_T.csv', '../Test_615B/Outputs/T615B_T.csv']

# Initialize lists to store results
Q_UPST_EXP = []
Q_DOWN_EXP = []
T_UPST_EXP = []
T_DOWN_EXP = []

# Loop through the files
for flow_file, temp_file in zip(files_flow, files_temp):
    # Read the data
    flow_data = pd.read_csv(flow_file)
    temp_data = pd.read_csv(temp_file)

    # Calculate upstream and downstream flow averages
    Q_UPST_EXP.append(flow_data.loc[flow_data['distance_m'] < 605, 'flow'].mean())
    Q_DOWN_EXP.append(flow_data.loc[flow_data['distance_m'] > 605, 'flow'].mean())

    # Calculate upstream and downstream temperature averages
    T_UPST_EXP.append(temp_data.loc[temp_data['distance_m'] < 605, 'Temp_C'].mean())
    T_DOWN_EXP.append(temp_data.loc[temp_data['distance_m'] > 605, 'Temp_C'].mean())

Q_UPST_EXP = [value for value in Q_UPST_EXP for _ in range(6)]
Q_DOWN_EXP = [value for value in Q_DOWN_EXP for _ in range(6)]
T_UPST_EXP = [value for value in T_UPST_EXP for _ in range(6)]
T_DOWN_EXP = [value for value in T_DOWN_EXP for _ in range(6)]
HRR = [10, 14, 49, 103]
HRR = [value for value in HRR for _ in range(6)]
test= ['T606A', 'T607', 'T611', 'T615B']
  
  
RES_Q_UPST = [x - y for x, y in zip(Q_UPST_EXP, Q_UPST_SES)]    
RES_Q_DOWN = [x - y for x, y in zip(Q_DOWN_EXP, Q_DOWN_SES)]

RES_TEMP_UPST = [x - y for x, y in zip(T_UPST_EXP, T_UPST_SES)]
RES_TEMP_DOWN = [x - y for x, y in zip(T_DOWN_EXP, T_DOWN_SES)]

##################### UPSTREAM FLOW ANALYSIS ################################################################################
############################################################################################################################
############################################################################################################################

Q_UPST_SES = np.array(Q_UPST_SES)
Q_UPST_EXP = np.array(Q_UPST_EXP)

############# Using All Average Values with Eficciency of 75% and 83% ######################################################

indices = [ 3,9,15,21]

Q_UPST_EXP_ALL = Q_UPST_EXP[indices]
Q_UPST_SES_ALL = Q_UPST_SES[indices]


mae = mean_absolute_error(Q_UPST_SES_ALL, Q_UPST_SES_ALL)
rmse = np.sqrt(mean_squared_error(Q_UPST_EXP_ALL, Q_UPST_SES_ALL))
r2 = r2_score(Q_UPST_EXP[indices], Q_UPST_SES_ALL)
error_percentage = np.abs(Q_UPST_EXP_ALL - Q_UPST_SES_ALL) / Q_UPST_EXP_ALL * 100
avg_error = np.mean(error_percentage)
RES_Q_UPST = [(x - y) for x, y in zip(Q_UPST_EXP_ALL, Q_UPST_SES_ALL)]    
mean = np.mean(RES_Q_UPST)
std = np.std(RES_Q_UPST)

############# Using All Average Values with Eficciency of 75% and 83% #####################################################

indices = [ 2, 8, 14, 20]

Q_UPST_EXP_75 = Q_UPST_EXP[indices]
Q_UPST_SES_75 = Q_UPST_SES[indices]

mae_75 = mean_absolute_error(Q_UPST_EXP[indices], Q_UPST_SES[indices])
rmse_75 = np.sqrt(mean_squared_error(Q_UPST_EXP[indices], Q_UPST_SES[indices]))
r2_75 = r2_score(Q_UPST_EXP[indices], Q_UPST_SES[indices])
error_percentage_75 = np.abs(Q_UPST_EXP[indices] - Q_UPST_SES[indices]) / Q_UPST_EXP[indices] * 100
avg_error_75 = np.mean(error_percentage_75)
RES_Q_UPST_75 = [(x - y) for x, y in zip(Q_UPST_EXP_75, Q_UPST_SES_75)]    
mean_75 = np.mean(RES_Q_UPST_75)
std_75 = np.std(RES_Q_UPST_75)

# Plot Q_UPST_EXP vs Q_UPST_SES

plt.figure(figsize=(8, 6))

highlight_indices = np.arange(0, len(Q_UPST_EXP), 6) 
plt.scatter(Q_UPST_EXP[highlight_indices], Q_UPST_SES[highlight_indices], marker='^', label='MIN-75%', zorder=5, facecolors='none', edgecolor='grey', alpha=0.7, s=90)

highlight_indices = np.arange(1, len(Q_UPST_EXP), 6) 
plt.scatter(Q_UPST_EXP[highlight_indices], Q_UPST_SES[highlight_indices], marker='D', label='MIN-83%', zorder=5, facecolors='none', edgecolor='grey', alpha=0.7, s=90)

highlight_indices = np.arange(2, len(Q_UPST_EXP), 6)  # Indices 2, 8, 14, 20, etc.
plt.scatter(Q_UPST_EXP[highlight_indices], Q_UPST_SES[highlight_indices], marker='s', label='AVE-75%', zorder=5, facecolors='blue', edgecolor='blue', s=110)

highlight_indices = np.arange(3, len(Q_UPST_EXP), 6)  # Indices 4, 10, 16, 22, etc.
plt.scatter(Q_UPST_EXP[highlight_indices], Q_UPST_SES[highlight_indices],  marker='v', label='AVE-83%', zorder=5, facecolors='green',  edgecolor='green', s=110)

highlight_indices = np.arange(4, len(Q_UPST_EXP), 6) 
plt.scatter(Q_UPST_EXP[highlight_indices], Q_UPST_SES[highlight_indices], marker='p', label='MAX-75%', zorder=5, facecolors='none', edgecolor='grey', alpha=0.7, s=90)

highlight_indices = np.arange(5, len(Q_UPST_EXP), 6) 
plt.scatter(Q_UPST_EXP[highlight_indices], Q_UPST_SES[highlight_indices], marker='H', label='MAX-83%', zorder=5, facecolors='none', edgecolor='grey', alpha=0.7, s=90)


# Add text annotations for each fire test
plt.annotate('T606A', xy=(Q_UPST_EXP[0], Q_UPST_SES[0]), xytext=(Q_UPST_EXP[0]-1.2, Q_UPST_SES[0] + 16),
              fontsize=10, color='black')

plt.annotate('T607', xy=(Q_UPST_EXP[6], Q_UPST_SES[6]), xytext=(Q_UPST_EXP[6] -1.2, Q_UPST_SES[6] +18),
              fontsize=10, color='black')

plt.annotate('T611', xy=(Q_UPST_EXP[12], Q_UPST_SES[12]), xytext=(Q_UPST_EXP[12] -1.2, Q_UPST_SES[12] +7),
              fontsize=10, color='black')

plt.annotate('T615B', xy=(Q_UPST_EXP[18], Q_UPST_SES[18]), xytext=(Q_UPST_EXP[18] -1.2 , Q_UPST_SES[18] +22),
             fontsize=10, color='black')

plt.gca().text(0.92, 0.2, 'AVE-75%-Values', fontsize=10, transform=plt.gca().transAxes, verticalalignment='bottom',
               horizontalalignment='right')
plt.gca().text(0.95, 0.05, f'Avg Error: {avg_error_75:.1f}%\nMean Error: {mean_75:.2f} m³/s \nStd Dev: {std_75:.2f} m³/s \nR²: {r2_75:.2f}',
               fontsize=10, transform=plt.gca().transAxes, verticalalignment='bottom',
               horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.7))

# Add second box with same values and title
plt.gca().text(0.675, 0.2, 'AVE-83%-Values', fontsize=10, transform=plt.gca().transAxes, verticalalignment='bottom',
               horizontalalignment='right')
plt.gca().text(0.705, 0.05, f'Avg Error: {avg_error:.1f}%\nMean Error: {mean:.2f} m³/s \nStd Dev: {std:.2f} m³/s \nR²: {r2:.2f}',
               fontsize=10, transform=plt.gca().transAxes, verticalalignment='bottom',
               horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.7))

plt.plot([min(Q_UPST_EXP), max(Q_UPST_EXP)], [min(Q_UPST_EXP), max(Q_UPST_EXP)], color='grey', linestyle='--', label='1:1 Line')
plt.xlabel('Measured Flowrate (m³/s)', fontsize=16)
plt.ylabel('Predicted Flowrate (m³/s)', fontsize=16)
plt.title('Average Upstream Flowrate', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.savefig('Average_Upstream_Flowrate.png', dpi=300) 
plt.show()



##################### DOWNSTREAM FLOW ANALYSIS ################################################################################
############################################################################################################################
############################################################################################################################

Q_DOWN_SES = np.array(Q_DOWN_SES)
Q_DOWN_EXP = np.array(Q_DOWN_EXP)

############# Using All Average Values with Eficciency of 75% and 83% ######################################################

indices = [3,9,15,21]

mae = mean_absolute_error(Q_DOWN_EXP[indices], Q_DOWN_SES[indices])
rmse = np.sqrt(mean_squared_error(Q_DOWN_EXP, Q_DOWN_SES))
r2 = r2_score(Q_DOWN_EXP[indices], Q_DOWN_SES[indices])
error_percentage = (Q_DOWN_EXP[indices] - Q_DOWN_SES[indices]) / Q_DOWN_EXP[indices] * 100
avg_error = np.mean(error_percentage)
RES_Q_DOWN = [(x - y) for x, y in zip(Q_DOWN_EXP, Q_DOWN_SES)]   
mean = np.mean(RES_Q_DOWN)
std = np.std(RES_Q_DOWN)

############# Using All Average Values with Eficciency of 75% and 83% #####################################################

indices = [ 2, 8, 14, 20]

Q_DOWN_EXP_75 = Q_DOWN_EXP[indices]
Q_DOWN_SES_75 = Q_DOWN_SES[indices]

mae_75 = mean_absolute_error(Q_DOWN_EXP[indices], Q_DOWN_SES[indices])
rmse_75 = np.sqrt(mean_squared_error(Q_DOWN_EXP[indices], Q_DOWN_SES[indices]))
r2_75 = r2_score(Q_DOWN_EXP[indices], Q_DOWN_SES[indices])
error_percentage_75 = (Q_DOWN_EXP[indices] - Q_DOWN_SES[indices]) / Q_DOWN_EXP[indices] * 100
avg_error_75 = np.mean(error_percentage_75)
RES_Q_DOWN_75 = [(x - y) for x, y in zip(Q_DOWN_EXP_75, Q_DOWN_SES_75)]    
mean_75 = np.mean(RES_Q_DOWN_75)
std_75 = np.std(RES_Q_DOWN_75)


plt.figure(figsize=(8, 6))

highlight_indices = np.arange(0, len(Q_DOWN_EXP), 6) 
plt.scatter(Q_DOWN_EXP[highlight_indices], Q_DOWN_SES[highlight_indices], marker='^', label='MIN-75%', zorder=5, facecolors='none', edgecolor='grey', alpha=0.7, s=90)

highlight_indices = np.arange(1, len(Q_DOWN_EXP), 6) 
plt.scatter(Q_DOWN_EXP[highlight_indices], Q_DOWN_SES[highlight_indices], marker='D', label='MIN-83%', zorder=5, facecolors='none', edgecolor='grey', alpha=0.7, s=90)

highlight_indices = np.arange(2, len(Q_DOWN_EXP), 6)  # Indices 2, 8, 14, 20, etc.
plt.scatter(Q_DOWN_EXP[highlight_indices], Q_DOWN_SES[highlight_indices], marker='s', label='AVE-75%', zorder=5, facecolors='blue', edgecolor='blue', s=110)

highlight_indices = np.arange(3, len(Q_DOWN_EXP), 6)  # Indices 4, 10, 16, 22, etc.
plt.scatter(Q_DOWN_EXP[highlight_indices], Q_DOWN_SES[highlight_indices],  marker='v', label='AVE-83%', zorder=5, facecolors='green',  edgecolor='green', s=110)

highlight_indices = np.arange(4, len(Q_DOWN_EXP), 6) 
plt.scatter(Q_DOWN_EXP[highlight_indices], Q_DOWN_SES[highlight_indices], marker='p', label='MAX-75%', zorder=5, facecolors='none', edgecolor='grey', alpha=0.7, s=90)

highlight_indices = np.arange(5, len(Q_DOWN_EXP), 6) 
plt.scatter(Q_DOWN_EXP[highlight_indices], Q_DOWN_SES[highlight_indices], marker='H', label='MAX-83%', zorder=5, facecolors='none', edgecolor='grey', alpha=0.7, s=90)


# Add text annotations for each fire test
plt.annotate('T606A', xy=(Q_DOWN_EXP[0], Q_DOWN_SES[0]), xytext=(Q_DOWN_EXP[0]-5, Q_DOWN_SES[0] + 33),
              fontsize=10, color='black')

plt.annotate('T607', xy=(Q_DOWN_EXP[6], Q_DOWN_SES[6]), xytext=(Q_DOWN_EXP[6] -5, Q_DOWN_SES[6] +40),
              fontsize=10, color='black')

plt.annotate('T611', xy=(Q_DOWN_EXP[12], Q_DOWN_SES[12]), xytext=(Q_DOWN_EXP[12] -5, Q_DOWN_SES[12] +100),
              fontsize=10, color='black')

plt.annotate('T615B', xy=(Q_DOWN_EXP[18], Q_DOWN_SES[18]), xytext=(Q_DOWN_EXP[18] -25, Q_DOWN_SES[18] +40),
             fontsize=10, color='black')

plt.gca().text(0.92, 0.2, 'AVE-75%-Values', fontsize=10, transform=plt.gca().transAxes, verticalalignment='bottom',
               horizontalalignment='right')
plt.gca().text(0.95, 0.05, f'Avg Error: {avg_error_75:.1f}%\nMean Error: {mean_75:.2f} m³/s \nStd Dev: {std_75:.2f} m³/s \nR²: {r2_75:.2f}',
               fontsize=10, transform=plt.gca().transAxes, verticalalignment='bottom',
               horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.7))

# Add second box with same values and title
plt.gca().text(0.67, 0.2, 'AVE-83%-Values', fontsize=10, transform=plt.gca().transAxes, verticalalignment='bottom',
               horizontalalignment='right')
plt.gca().text(0.7, 0.05, f'Avg Error: {avg_error:.1f}%\nMean Error: {mean:.2f} m³/s \nStd Dev: {std:.2f} m³/s \nR²: {r2:.2f}',
               fontsize=10, transform=plt.gca().transAxes, verticalalignment='bottom',
               horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.7))

plt.plot([min(Q_DOWN_EXP), max(Q_DOWN_EXP)], [min(Q_DOWN_EXP), max(Q_DOWN_EXP)], color='grey', linestyle='--', label='1:1 Line')
plt.xlabel('Measured Flowrate (m³/s)', fontsize=16)
plt.ylabel('Predicted Flowrate (m³/s)', fontsize=16)
plt.title('Average Downstream Flowrate (m³/s)', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.savefig('Average_Downstream_Flowrate.png', dpi=300) 
plt.show()

##################### DOWNSTREAM TEMPERATURE ANALYSIS ################################################################################
############################################################################################################################
############################################################################################################################

T_DOWN_SES = np.array(T_DOWN_SES)
T_DOWN_EXP = np.array(T_DOWN_EXP)

############# Using All Average Values with Eficciency of 75% and 83% ######################################################

indices = [ 3,9,15,21]

mae = mean_absolute_error(T_DOWN_EXP[indices], T_DOWN_SES[indices])
rmse = np.sqrt(mean_squared_error(T_DOWN_EXP, T_DOWN_SES))
r2 = r2_score(T_DOWN_EXP[indices], T_DOWN_SES[indices])
error_percentage = (T_DOWN_EXP[indices] - T_DOWN_SES[indices]) / T_DOWN_EXP[indices] * 100
avg_error = np.mean(error_percentage)
RES_T_DOWN = [(x - y) for x, y in zip(T_DOWN_EXP, T_DOWN_SES)]   
mean = np.mean(RES_T_DOWN)
std = np.std(RES_T_DOWN)

############# Using All Average Values with Eficciency of 75% and 83% #####################################################

indices = [ 2, 8, 14, 20]

T_DOWN_EXP_75 = T_DOWN_EXP[indices]
T_DOWN_SES_75 = T_DOWN_SES[indices]

mae_75 = mean_absolute_error(T_DOWN_EXP[indices], T_DOWN_SES[indices])
rmse_75 = np.sqrt(mean_squared_error(T_DOWN_EXP[indices], T_DOWN_SES[indices]))
r2_75 = r2_score(T_DOWN_EXP[indices], T_DOWN_SES[indices])
error_percentage_75 = (T_DOWN_EXP[indices] - T_DOWN_SES[indices]) / T_DOWN_EXP[indices] * 100
avg_error_75 = np.mean(error_percentage_75)
RES_T_DOWN_75 = [(x - y) for x, y in zip(T_DOWN_EXP_75, T_DOWN_SES_75)]    
mean_75 = np.mean(RES_T_DOWN_75)
std_75 = np.std(RES_T_DOWN_75)


plt.figure(figsize=(8, 6))

highlight_indices = np.arange(0, len(T_DOWN_EXP), 6) 
plt.scatter(T_DOWN_EXP[highlight_indices], T_DOWN_SES[highlight_indices], marker='^', label='MIN-75%', zorder=5, facecolors='none', edgecolor='grey', alpha=0.7, s=90)

highlight_indices = np.arange(1, len(T_DOWN_EXP), 6) 
plt.scatter(T_DOWN_EXP[highlight_indices], T_DOWN_SES[highlight_indices], marker='D', label='MIN-83%', zorder=5, facecolors='none', edgecolor='grey', alpha=0.7, s=90)

highlight_indices = np.arange(2, len(T_DOWN_EXP), 6)  # Indices 2, 8, 14, 20, etc.
plt.scatter(T_DOWN_EXP[highlight_indices], T_DOWN_SES[highlight_indices], marker='s', label='AVE-75%', zorder=5, facecolors='blue', edgecolor='blue', s=110)

highlight_indices = np.arange(3, len(T_DOWN_EXP), 6)  # Indices 4, 10, 16, 22, etc.
plt.scatter(T_DOWN_EXP[highlight_indices], T_DOWN_SES[highlight_indices],  marker='v', label='AVE-83%', zorder=5, facecolors='green',  edgecolor='green', s=110)

highlight_indices = np.arange(4, len(T_DOWN_EXP), 6) 
plt.scatter(T_DOWN_EXP[highlight_indices], T_DOWN_SES[highlight_indices], marker='p', label='MAX-75%', zorder=5, facecolors='none', edgecolor='grey', alpha=0.7, s=90)

highlight_indices = np.arange(5, len(T_DOWN_EXP), 6) 
plt.scatter(T_DOWN_EXP[highlight_indices], T_DOWN_SES[highlight_indices], marker='H', label='MAX-83%', zorder=5, facecolors='none', edgecolor='grey', alpha=0.7, s=90)

# Add text annotations for each fire test
plt.annotate('T606A', xy=(T_DOWN_EXP[0], T_DOWN_SES[0]), xytext=(T_DOWN_EXP[0]-10, T_DOWN_SES[0] + 55),
              fontsize=10, color='black')

plt.annotate('T607', xy=(T_DOWN_EXP[6], T_DOWN_SES[6]), xytext=(T_DOWN_EXP[6] - 10, T_DOWN_SES[6] + 35),
              fontsize=10, color='black')

plt.annotate('T611', xy=(T_DOWN_EXP[12], T_DOWN_SES[12]), xytext=(T_DOWN_EXP[12] - 10, T_DOWN_SES[12] +120),
              fontsize=10, color='black')

plt.annotate('T615B', xy=(T_DOWN_EXP[18], T_DOWN_SES[18]), xytext=(T_DOWN_EXP[18] -30, T_DOWN_SES[18] +40),
             fontsize=10, color='black')

plt.gca().text(0.92, 0.2, 'AVE-75%-Values', fontsize=10, transform=plt.gca().transAxes, verticalalignment='bottom',
               horizontalalignment='right')
plt.gca().text(0.95, 0.05, f'Avg Error: {avg_error_75:.1f}%\nMean Error: {mean_75:.2f} °C \nStd Dev: {std_75:.2f} °C \nR²: {r2_75:.2f}',
               fontsize=10, transform=plt.gca().transAxes, verticalalignment='bottom',
               horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.7))

# Add second box with same values and title
plt.gca().text(0.67, 0.2, 'AVE-83%-Values', fontsize=10, transform=plt.gca().transAxes, verticalalignment='bottom',
               horizontalalignment='right')
plt.gca().text(0.7, 0.05, f'Avg Error: {avg_error:.1f}%\nMean Error: {mean:.2f} °C \nStd Dev: {std:.2f} °C \nR²: {r2:.2f}',
               fontsize=10, transform=plt.gca().transAxes, verticalalignment='bottom',
               horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.7))

plt.plot([min(T_DOWN_EXP), max(T_DOWN_EXP)], [min(T_DOWN_EXP), max(T_DOWN_EXP)], color='grey', linestyle='--', label='1:1 Line')
plt.xlabel('Measured Temperature (°C)', fontsize=16)
plt.ylabel('Predicted Temperature (°C)', fontsize=16)
plt.title('Average Downstream Temperature (°C)', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.savefig('Average_Downstream_Temperature.png', dpi=300) 
plt.show()

##################### UPSTREAM TEMPERATURE ANALYSIS ################################################################################
############################################################################################################################
############################################################################################################################

T_UPST_SES = np.array(T_UPST_SES)
T_UPST_EXP = np.array(T_UPST_EXP)

############# Using All Average Values with Eficciency of 75% and 83% ######################################################

indices = [ 3,9,15,21]

T_UPST_EXP_ALL = T_UPST_EXP[indices]
T_UPST_SES_ALL = T_UPST_SES[indices]

mae = mean_absolute_error(T_UPST_EXP_ALL, T_UPST_SES_ALL)
rmse = np.sqrt(mean_squared_error(T_UPST_EXP_ALL, T_UPST_SES_ALL))
r2 = -r2_score(T_UPST_EXP[indices], T_UPST_SES[indices])/100
error_percentage = (T_UPST_EXP[indices] - T_UPST_SES[indices]) / T_UPST_EXP[indices] * 100
avg_error = np.mean(error_percentage)
RES_T_UPST = [(x - y) for x, y in zip(T_UPST_EXP_ALL, T_UPST_SES_ALL)]    
mean = np.mean(RES_T_UPST)
std = np.std(RES_T_UPST)

############# Using All Average Values with Eficciency of 75% and 83% #####################################################

indices = [ 2, 8, 14, 20]

T_UPST_EXP_75 = T_UPST_EXP[indices]
T_UPST_SES_75 = T_UPST_SES[indices]

mae_75 = mean_absolute_error(T_UPST_EXP[indices], T_UPST_SES[indices])
rmse_75 = np.sqrt(mean_squared_error(T_UPST_EXP[indices], T_UPST_SES[indices]))
r2_75 = -r2_score(T_UPST_EXP[indices], T_UPST_SES[indices])/100
error_percentage_75 = (T_UPST_EXP[indices] - T_UPST_SES[indices]) / T_UPST_EXP[indices] * 100
avg_error_75 = np.mean(error_percentage_75)
RES_T_UPST_75 = [(x - y) for x, y in zip(T_UPST_EXP_75, T_UPST_SES_75)]    
mean_75 = np.mean(RES_T_UPST_75)
std_75 = np.std(RES_T_UPST_75)


plt.figure(figsize=(8, 6))

highlight_indices = np.arange(0, len(T_UPST_EXP), 6) 
plt.scatter(T_UPST_EXP[highlight_indices], T_UPST_SES[highlight_indices], marker='^', label='MIN-75%', zorder=5, facecolors='none', edgecolor='grey', alpha=0.7, s=90)

highlight_indices = np.arange(1, len(T_UPST_EXP), 6) 
plt.scatter(T_UPST_EXP[highlight_indices], T_UPST_SES[highlight_indices], marker='D', label='MIN-83%', zorder=5, facecolors='none', edgecolor='grey', alpha=0.7, s=90)

highlight_indices = np.arange(2, len(T_UPST_EXP), 6)  # Indices 2, 8, 14, 20, etc.
plt.scatter(T_UPST_EXP[highlight_indices], T_UPST_SES[highlight_indices], marker='s', label='AVE-75%', zorder=5, facecolors='blue', edgecolor='blue', s=110)

highlight_indices = np.arange(3, len(T_UPST_EXP), 6)  # Indices 4, 10, 16, 22, etc.
plt.scatter(T_UPST_EXP[highlight_indices], T_UPST_SES[highlight_indices],  marker='v', label='AVE-83%', zorder=5, facecolors='green',  edgecolor='green', s=110)

highlight_indices = np.arange(4, len(T_UPST_EXP), 6) 
plt.scatter(T_UPST_EXP[highlight_indices], T_UPST_SES[highlight_indices], marker='p', label='MAX-75%', zorder=5, facecolors='none', edgecolor='grey', alpha=0.7, s=90)

highlight_indices = np.arange(5, len(T_UPST_EXP), 6) 
plt.scatter(T_UPST_EXP[highlight_indices], T_UPST_SES[highlight_indices], marker='H', label='MAX-83%', zorder=5, facecolors='none', edgecolor='grey', alpha=0.7, s=90)


plt.gca().text(0.25, 0.9, 'AVE-75%-Values', fontsize=10, transform=plt.gca().transAxes, verticalalignment='bottom',
               horizontalalignment='right')
plt.gca().text(0.26, 0.75, f'Avg Error: {avg_error_75:.1f}%\nMean Error: {mean_75:.2f} °C \nStd Dev: {std_75:.2f} °C \nR²: {r2_75:.2f}',
               fontsize=10, transform=plt.gca().transAxes, verticalalignment='bottom',
               horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.7))

# Add second box with same values and title
plt.gca().text(0.45, 0.9, 'AVE-83%-Values', fontsize=10, transform=plt.gca().transAxes, verticalalignment='bottom',
               horizontalalignment='right')
plt.gca().text(0.48, 0.75, f'Avg Error: {avg_error_75:.1f}%\nMean Error: {mean_75:.2f} °C \nStd Dev: {std_75:.2f} °C \nR²: {r2_75:.2f}',
               fontsize=10, transform=plt.gca().transAxes, verticalalignment='bottom',
               horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.7))

plt.plot([min(T_UPST_EXP), max(T_UPST_EXP)], [min(T_UPST_EXP), max(T_UPST_EXP)], color='grey', linestyle='--', label='1:1 Line')
plt.xlabel('Measured Temperature (°C)', fontsize=16)
plt.ylabel('Predicted Temperature (°C)', fontsize=16)
plt.title('Average Upstream Temperature (°C)', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.ylim(0,40)
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.savefig('Average_Upstream_Temperature.png', dpi=300) 
plt.show()