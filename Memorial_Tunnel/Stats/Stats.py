# -*- coding: utf-8 -*-
"""
Created on Tue May  6 08:37:58 2025

@author: mdaza
"""

import pandas as pd
import matplotlib.pyplot as plt
import glob
import itertools

markers = ['s', 'o', '^', 'D', 'P']  # square, circle, triangle, diamond, plus-filled

##################### PREDICTED SES FLOWS ##################################################################################
############################################################################################################################
############################################################################################################################

SES606A = pd.read_csv('../Test_606A/Outputs/SES_results_606A.csv')
SES607_14MW = pd.read_csv('../Test_607/Outputs/SES_results_607_14MW.csv')
# SES607_20MW = pd.read_csv('../Test_607/Outputs/SES_results_607_20MW.csv')
SES611 = pd.read_csv('../Test_611/Outputs/SES_results_611.csv')
SES615B = pd.read_csv('../Test_615B/Outputs/SES_results_615B.csv')

HRR = [10, 14, 49, 103]
test= ['T606A', 'T607', 'T611', 'T615B']

Q_UPST_SES606A = SES606A.loc[SES606A['length'] < 605, 'flowrate'].mean()
Q_UPST_SES607_14MW = SES607_14MW.loc[SES607_14MW['length'] < 605, 'flowrate'].mean()
# Q_UPST_SES607_20MW = SES607_20MW.loc[SES607_20MW['length'] < 605, 'flowrate'].mean()
Q_UPST_SES611 = SES611.loc[SES611['length'] < 605, 'flowrate'].mean()
Q_UPST_SES615B = SES615B.loc[SES615B['length'] < 605, 'flowrate'].mean()

Q_UPST_SES = [Q_UPST_SES606A, Q_UPST_SES607_14MW,  Q_UPST_SES611, Q_UPST_SES615B]

Q_DOWN_SES606A = SES606A.loc[SES606A['length'] > 605, 'flowrate'].mean()
Q_DOWN_SES607_14MW = SES607_14MW.loc[SES607_14MW['length'] > 605, 'flowrate'].mean()
# Q_DOWN_SES607_20MW = SES607_20MW.loc[SES607_20MW['length'] > 605, 'flowrate'].mean()
Q_DOWN_SES611 = SES611.loc[SES611['length'] > 605, 'flowrate'].mean()
Q_DOWN_SES615B = SES615B.loc[SES615B['length'] > 605, 'flowrate'].mean()

Q_DOWN_SES = [Q_DOWN_SES606A, Q_DOWN_SES607_14MW,  Q_DOWN_SES611, Q_DOWN_SES615B]

##################### PREDICTED MEASURED FLOWS #############################################################################
EXP606A = pd.read_csv('../Test_606A/Outputs/T606A_Q.csv')
EXP607 = pd.read_csv('../Test_607/Outputs/T607_Q.csv')
EXP611 = pd.read_csv('../Test_611/Outputs/T611_Q.csv')
EXP615B = pd.read_csv('../Test_615B/Outputs/T615B_Q.csv')

Q_UPST_EXP606A = EXP606A.loc[EXP606A['distance_m'] < 605, 'flow'].mean()
Q_UPST_EXP607 = EXP607.loc[EXP607['distance_m'] < 605, 'flow'].mean()
Q_UPST_EXP611 = EXP611.loc[EXP611['distance_m'] < 605, 'flow'].mean()
Q_UPST_EXP615B = EXP615B.loc[EXP611['distance_m'] < 605, 'flow'].mean()

Q_UPST_EXP = [Q_UPST_EXP606A,  Q_UPST_EXP607, Q_UPST_EXP611, Q_UPST_EXP615B]

RES_Q_UPST_EXP = [x - y for x, y in zip(Q_UPST_EXP, Q_UPST_SES)]

Q_DOWN_EXP606A = EXP606A.loc[EXP606A['distance_m'] > 605, 'flow'].mean()
Q_DOWN_EXP607 = EXP607.loc[EXP607['distance_m'] > 605, 'flow'].mean()
Q_DOWN_EXP611 = EXP611.loc[EXP611['distance_m'] > 605, 'flow'].mean()
Q_DOWN_EXP615B = EXP615B.loc[EXP611['distance_m'] > 605, 'flow'].mean()

Q_DOWN_EXP = [Q_DOWN_EXP606A,  Q_DOWN_EXP607, Q_DOWN_EXP611, Q_DOWN_EXP615B]

RES_Q_DOWN_EXP = [x - y for x, y in zip(Q_DOWN_EXP, Q_DOWN_SES)]

##################### UPSTREAM FLOW ANALYSIS ################################################################################

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

Q_UPST_SES = np.array(Q_UPST_SES)
Q_UPST_EXP = np.array(Q_UPST_EXP)

mae = mean_absolute_error(Q_UPST_EXP, Q_UPST_SES)
rmse = np.sqrt(mean_squared_error(Q_UPST_EXP, Q_UPST_SES))
r2 = r2_score(Q_UPST_EXP, Q_UPST_SES)
error_percentage = np.abs(Q_UPST_EXP - Q_UPST_SES) / Q_UPST_EXP * 100
avg_error = np.mean(error_percentage)
mean = np.mean(RES_Q_UPST_EXP)
std = np.std(RES_Q_UPST_EXP)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Coefficient of Determination (R²): {r2}")

plt.scatter(HRR, RES_Q_UPST_EXP)
plt.axhline(y=0, color='gray', linestyle='--')  # add horizontal line at y = 0
plt.ylim([-50, 50])
plt.legend(fontsize=6)
plt.xlabel('HRR (MW)', fontsize=12)
plt.ylabel('Residual (m3/s)', fontsize=12)
plt.title('Average Flowrate upstream', fontsize=12)
plt.tight_layout()
plt.show()

plt.figure()
for i in range(len(Q_UPST_SES)):
    plt.plot(Q_UPST_EXP[i], Q_UPST_SES[i], marker=markers[i % len(markers)], linestyle='None', color='black', markerfacecolor='none', label=f'{test[i]} ({HRR[i]} MW)')
    
plt.gca().text(0.95, 0.05, f'Avg Error: {avg_error:.1f}%\nMean Error: {mean:.2f} m³/s \nStd Dev: {std:.2f} m³/s',
               fontsize=10, transform=plt.gca().transAxes, verticalalignment='bottom',
               horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.7))

plt.plot([min(Q_UPST_EXP), max(Q_UPST_EXP)], [min(Q_UPST_EXP), max(Q_UPST_EXP)], '--', color='grey', label='1:1 Line')
plt.xlabel('Measured Flowrate (m³/s)', fontsize=12)
plt.ylabel('Predicted Flowrate (m³/s)', fontsize=12)
plt.title('Average Upstream Flowrate', fontsize=12)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()

##################### DOWNSTREAM FLOW ANALYSIS ################################################################################

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

Q_DOWN_SES = np.array(Q_DOWN_SES)
Q_DOWN_EXP = np.array(Q_DOWN_EXP)

mae = mean_absolute_error(Q_DOWN_EXP, Q_DOWN_SES)
rmse = np.sqrt(mean_squared_error(Q_DOWN_EXP, Q_DOWN_SES))
r2 = r2_score(Q_DOWN_EXP, Q_DOWN_SES)
error_percentage = np.abs(Q_DOWN_EXP - Q_DOWN_SES) / Q_DOWN_EXP * 100
avg_error = np.mean(error_percentage)
mean = np.mean(RES_Q_DOWN_EXP)
std = np.std(RES_Q_DOWN_EXP)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Coefficient of Determination (R²): {r2}")

plt.scatter(HRR, RES_Q_DOWN_EXP)
plt.axhline(y=0, color='gray', linestyle='--')  # add horizontal line at y = 0
plt.ylim([-50, 50])
plt.legend(fontsize=8)
plt.xlabel('HRR (MW)', fontsize=12)
plt.ylabel('Residual (m3/s)', fontsize=12)
plt.title('Average Flowrate downstream', fontsize=12)
plt.tight_layout()
plt.show()

plt.figure()
for i in range(len(Q_DOWN_SES)):
    plt.plot(Q_DOWN_EXP[i], Q_DOWN_SES[i], marker=markers[i % len(markers)], linestyle='None', color='black', markerfacecolor='none', label=f'{test[i]} ({HRR[i]} MW)')
    
plt.gca().text(0.95, 0.05, f'Avg Error: {avg_error:.1f}%\nMean Error: {mean:.2f} m³/s \nStd Dev: {std:.2f} m³/s',
               fontsize=10, transform=plt.gca().transAxes, verticalalignment='bottom',
               horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.7))

plt.plot([min(Q_DOWN_EXP), max(Q_DOWN_EXP)], [min(Q_DOWN_EXP), max(Q_DOWN_EXP)], '--', color='grey', label='1:1 Line')
plt.xlabel('Measured Flowrate (m³/s)', fontsize=12)
plt.ylabel('Predicted Flowrate (m³/s)', fontsize=12)
plt.title('Average Downstream Flowrate', fontsize=12)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()


###################################################################################################################################
###################################################################################################################################
##################### PREDICTED SES TEMPERATURES ##################################################################################

TEMP_UPST_SES606A = SES606A.loc[SES606A['length'] < 605, 'temperature'].mean()
TEMP_UPST_SES607_14MW = SES607_14MW.loc[SES607_14MW['length'] < 605, 'temperature'].mean()
# TEMP_UPST_SES607_20MW = SES607_20MW.loc[SES607_20MW['length'] < 605, 'temperature'].mean()
TEMP_UPST_SES611 = SES611.loc[SES611['length'] < 605, 'temperature'].mean()
TEMP_UPST_SES615B = SES615B.loc[SES615B['length'] < 605, 'temperature'].mean()

TEMP_UPST_SES = [TEMP_UPST_SES606A, TEMP_UPST_SES607_14MW, TEMP_UPST_SES611, TEMP_UPST_SES615B]

TEMP_DOWN_SES606A = SES606A.loc[SES606A['length'] > 605, 'temperature'].mean()
TEMP_DOWN_SES607_14MW = SES607_14MW.loc[SES607_14MW['length'] > 605, 'temperature'].mean()
# TEMP_DOWN_SES607_20MW = SES607_20MW.loc[SES607_20MW['length'] > 605, 'temperature'].mean()
TEMP_DOWN_SES611 = SES611.loc[SES611['length'] > 605, 'temperature'].mean()
TEMP_DOWN_SES615B = SES615B.loc[SES615B['length'] > 605, 'temperature'].mean()

TEMP_DOWN_SES = [TEMP_DOWN_SES606A, TEMP_DOWN_SES607_14MW, TEMP_DOWN_SES611, TEMP_DOWN_SES615B]

##################### PREDICTED MEASURED FLOWS #############################################################################

TEMP_EXP606A = pd.read_csv('../Test_606A/Outputs/T606A_T.csv')
TEMP_EXP607 = pd.read_csv('../Test_607/Outputs/T607_T.csv')
TEMP_EXP611 = pd.read_csv('../Test_611/Outputs/T611_T.csv')
TEMP_EXP615B = pd.read_csv('../Test_615B/Outputs/T615B_T.csv')

TEMP_UPST_EXP606A = TEMP_EXP606A.loc[TEMP_EXP606A['distance_m'] < 605, 'Temp_C'].mean()
TEMP_UPST_EXP607 = TEMP_EXP607.loc[TEMP_EXP607['distance_m'] < 605, 'Temp_C'].mean()
TEMP_UPST_EXP611 = TEMP_EXP611.loc[TEMP_EXP611['distance_m'] < 605, 'Temp_C'].mean()
TEMP_UPST_EXP615B = TEMP_EXP615B.loc[TEMP_EXP611['distance_m'] < 605, 'Temp_C'].mean()

TEMP_UPST_EXP = [TEMP_UPST_EXP606A, TEMP_UPST_EXP607, TEMP_UPST_EXP611, TEMP_UPST_EXP615B]

TEMP_DOWN_EXP606A = TEMP_EXP606A.loc[TEMP_EXP606A['distance_m'] > 605, 'Temp_C'].mean()
TEMP_DOWN_EXP607 = TEMP_EXP607.loc[TEMP_EXP607['distance_m'] > 605, 'Temp_C'].mean()
TEMP_DOWN_EXP611 = TEMP_EXP611.loc[TEMP_EXP611['distance_m'] > 605, 'Temp_C'].mean()
TEMP_DOWN_EXP615B = TEMP_EXP615B.loc[TEMP_EXP611['distance_m'] > 605, 'Temp_C'].mean()

TEMP_DOWN_EXP = [TEMP_DOWN_EXP606A, TEMP_DOWN_EXP607, TEMP_DOWN_EXP611, TEMP_DOWN_EXP615B]

RES_TEMP_UPST = [x - y for x, y in zip(TEMP_UPST_EXP, TEMP_UPST_SES)]

RES_TEMP_DOWN = [x - y for x, y in zip(TEMP_DOWN_EXP, TEMP_DOWN_SES)]


##################### UPSTREAM TEMPERATURE ANALYSIS ################################################################################

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

TEMP_UPST_SES = np.array(TEMP_UPST_SES)
TEMP_UPST_EXP = np.array(TEMP_UPST_EXP)

mae = mean_absolute_error(TEMP_UPST_EXP, TEMP_UPST_SES)
rmse = np.sqrt(mean_squared_error(TEMP_UPST_EXP, TEMP_UPST_SES))
r2 = r2_score(TEMP_UPST_EXP, TEMP_UPST_SES)
error_percentage = np.abs( TEMP_UPST_EXP - TEMP_UPST_SES) / TEMP_UPST_EXP * 100
avg_error = np.mean(error_percentage)
mean = np.mean(RES_TEMP_UPST)
std = np.std(RES_TEMP_UPST)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Coefficient of Determination (R²): {r2}")

plt.scatter(HRR, RES_TEMP_UPST)
plt.axhline(y=0, color='gray', linestyle='--')  # add horizontal line at y = 0
plt.ylim([-50, 50])
plt.legend(fontsize=10)
plt.xlabel('HRR (MW)', fontsize=16)
plt.ylabel('Residual (°C)', fontsize=16)
plt.title('Temperature upstream', fontsize=12)
plt.tight_layout()
plt.show()

plt.figure()
for i in range(len(Q_UPST_SES)):
    plt.plot(TEMP_UPST_EXP[i], TEMP_UPST_SES[i], marker=markers[i % len(markers)], linestyle='None', color='black', markerfacecolor='none', label=f'{test[i]} ({HRR[i]} MW)')
    
plt.gca().text(0.95, 0.95, f'Avg Error: {avg_error:.1f}%\nMean Error: {mean:.2f} °C \nStd Dev: {std:.2f} °C ',
               fontsize=10, transform=plt.gca().transAxes, verticalalignment='top',
               horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.7))

plt.plot([min(TEMP_UPST_EXP), max(TEMP_UPST_EXP)], [min(TEMP_UPST_EXP), max(TEMP_UPST_EXP)], '--', color='grey', label='1:1 Line')
plt.xlabel('Measured Temperature (°C)', fontsize=12)
plt.ylabel('Predicted Temperature (°C)', fontsize=12)
plt.ylim([0,40])
plt.title('Average Upstream Temperature', fontsize=12)
plt.legend(fontsize=10, loc='upper left')
plt.tight_layout()
plt.show()

##################### DOWNSTREAM TEMPERATURE ANALYSIS ################################################################################

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

TEMP_DOWN_SES = np.array(TEMP_DOWN_SES)
TEMP_DOWN_EXP = np.array(TEMP_DOWN_EXP)

mae = mean_absolute_error(TEMP_DOWN_EXP, TEMP_DOWN_SES)
rmse = np.sqrt(mean_squared_error(TEMP_DOWN_EXP, TEMP_DOWN_SES))
r2 = r2_score(TEMP_DOWN_EXP, TEMP_DOWN_SES)
error_percentage = np.abs(TEMP_DOWN_EXP - TEMP_DOWN_SES) / TEMP_DOWN_EXP * 100
avg_error = np.mean(error_percentage)
mean = np.mean(RES_TEMP_DOWN)
std = np.std(RES_TEMP_DOWN)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Coefficient of Determination (R²): {r2}")

plt.scatter(HRR, RES_TEMP_DOWN)
plt.axhline(y=0, color='gray', linestyle='--')  # add horizontal line at y = 0
plt.ylim([-50, 50])
plt.legend(fontsize=10)
plt.xlabel('HRR (MW)', fontsize=16)
plt.ylabel('Residual (m3/s)', fontsize=16)
plt.title('Average Temperature Downstream', fontsize=12)
plt.tight_layout()
plt.show()

plt.figure()
for i in range(len(Q_DOWN_SES)):
    plt.plot(TEMP_DOWN_EXP[i], TEMP_DOWN_SES[i], marker=markers[i % len(markers)], linestyle='None', color='black', markerfacecolor='none', label=f'{test[i]} ({HRR[i]} MW)')
    
plt.gca().text(0.95, 0.05, f'Avg Error: {avg_error:.1f}%\nMean Error: {mean:.2f} °C \nStd Dev: {std:.2f} °C ',
               fontsize=10, transform=plt.gca().transAxes, verticalalignment='bottom',
               horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.7))

plt.plot([min(TEMP_DOWN_EXP), max(TEMP_DOWN_EXP)], [min(TEMP_DOWN_EXP), max(TEMP_DOWN_EXP)], '--', color='grey', label='1:1 Line')
plt.xlabel('Measured Temperature (°C)', fontsize=12)
plt.ylabel('Predicted Temperature (°C)', fontsize=12)
plt.title('Average Downstream Temperature', fontsize=12)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()
