# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 10:25:09 2025

@author: mdaza
"""

import pandas as pd
import matplotlib.pyplot as plt
import glob
import itertools

#####################MEASURED HRR VALUES ########################

T607_HRR_data = pd.read_csv(
    './Measured_data/HRR607.csv', skiprows=[1])

T607_HRR = T607_HRR_data.iloc[:, [1,2]].astype(
    float).mean(axis=1)  # 15 MW used in SES

print(T607_HRR_data.iloc[28:32, 1:3].astype(
    float).mean(axis=0).mean(axis=0) ) # 15 MW used in SES


#####################MEASURED TEMPERATURE VALUES + AVERAGING TEMP IN EACH LOOP ########################


loops = [202, 301, 302, 303, 304, 205, 305,
         306, 307, 207, 208, 209, 211, 213, 214]
loops_length_ft = [2736, 2373, 2236, 2116, 2059, 2019,
                   1982, 1923, 1816, 1668, 1399, 1053, 692, 347, 65]
loops_length_m = [ft * 0.3048 for ft in loops_length_ft]
T607_Temp = pd.read_csv('./Measured_data/TP607.csv')
SES_segment_length = pd.read_csv('./Measured_data/SES_segment_length.csv')
T607_Temp = T607_Temp.drop(index=0)
loop_n = []
T607_data_Temp = []
a = 1
for loop in loops:
    loop_n.append(T607_Temp.keys().str.contains(str(loop)).sum())
for n in loop_n:
    T607_data_Temp.append(
        T607_Temp.iloc[28:32, a:n+a].astype(float).mean(axis=1).mean(axis=0))
    a += n
T607_measured_data_Temp = pd.DataFrame(
    {"loop": loops, "distance_m": loops_length_m, "Temp_C": T607_data_Temp})
T607_measured_data_Temp = T607_measured_data_Temp.sort_values(by=[
                                                              'distance_m'])
plt.figure()
plt.plot(T607_measured_data_Temp['distance_m'], T607_measured_data_Temp['Temp_C'], marker='o', color='black')
plt.xlabel('Distance (m)')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.show()

##################### MEASURED FLOW VALUES  #########################################################

loops = [214, 209, 208, 207, 307, 305, 304, 302, 301, 202]
loops_length_ft = [65, 1053, 1399, 1668, 1816, 1982, 2059, 2236, 2373, 2736]
loops_length_m = [ft * 0.3048 for ft in loops_length_ft]
Q607_Flow = pd.read_csv('./Measured_data/QP607.csv')
Q607_Flow = Q607_Flow.drop(index=0)
T607_measured_Flow = abs(Q607_Flow.iloc[28:32, 1:].astype(float).mean(axis=0)).tolist()
T607_measured_data_Flow = pd.DataFrame(
    {"loop": loops, "distance_m": loops_length_m, "flow": T607_measured_Flow})

plt.figure()
plt.plot(T607_measured_data_Flow['distance_m'], T607_measured_data_Flow['flow'], marker='o', color='black')
plt.xlabel('Distance (m)')
plt.ylabel('Flow (m³/s)')
plt.grid(True)
plt.show()

##################### SES RESULTS EXTRACTION  ###########################################################3

ses_result_files = ['./SES_results/MT-T607-R3.xlsx'] # './SES_results/MT-T607-R4.xlsx']
naming = ['SES']#, 'SES-20 MW']
colors = ['blue']#, 'orange']
all_results = []

for file in ses_result_files:
    data_t = pd.read_excel(file, sheet_name="Temperature")
    data_q = pd.read_excel(file, sheet_name="Flow Rate")
    temp = data_t.iloc[:, 65:106].mean(axis=1).tolist()
    flow = data_q.iloc[:, 65:106].mean(axis=1).tolist()
    pressure = 101325
    R_constant = 287.05
    density = [pressure / (R_constant * (273.15 + t)) for t in temp]
    densi_standard = 1.251842913
    flow2 = [f * densi_standard / dens for f, dens in itertools.zip_longest(flow, density)]
    SES_results = pd.DataFrame({
        "length": SES_segment_length['distance'].tolist(),
        "temperature": temp,
        "flowrate": flow2,
        "file": file
    })
    all_results.append(SES_results)

combined_results = pd.concat(all_results, ignore_index=True)

plt.figure()
a = 0
for file in ses_result_files:
    subset = combined_results[combined_results['file'] == file]
    plt.plot(subset["length"], subset["temperature"], color=colors[a], label=naming[a], marker="s", fillstyle='none')
    
    T607_Temp_avg_upstream = subset.loc[subset['length'] < 605, 'temperature'].mean()
    T607_Temp_avg_downstream = subset.loc[subset['length'] > 605, 'temperature'].mean()
    print(naming[a])
    print('predicted avg temperature upstream the fire in T607', T607_Temp_avg_upstream)
    print('predicted avg temperature downstream the fire in T607', T607_Temp_avg_downstream)
    a +=1
    
    
plt.plot(T607_measured_data_Temp["distance_m"], T607_measured_data_Temp["Temp_C"],
         label="Measured", marker="o", color='black', fillstyle='none')
plt.legend(fontsize=10)
plt.xlabel('Distance from North Portal (m)', fontsize=16)
plt.ylabel('Temperature(°C)', fontsize=16)
plt.title('Test 607 - Temperature Distribution', fontsize=12)
plt.tight_layout()
plt.show()

plt.figure()


b = 0
for file in ses_result_files:
    subset = combined_results[combined_results['file'] == file]
    plt.plot(subset["length"], subset["flowrate"], color=colors[b], label=naming[b], marker="s", fillstyle='none')
    
    T607_Flow_avg_upstream = subset.loc[subset['length'] < 605, 'flowrate'].mean()
    T607_Flow_avg_downstream = subset.loc[subset['length'] > 605, 'flowrate'].mean()
    print(naming[b])
    print('predicted avg flow upstream the fire in T607', T607_Flow_avg_upstream)
    print('predicted avg flow downstream the fire in T607', T607_Flow_avg_downstream)
    b += 1
    
plt.plot(T607_measured_data_Flow["distance_m"], T607_measured_data_Flow["flow"],
         label="Measured", marker="o", color='black', fillstyle='none')

T607_Flow_avg_upstream_mea = T607_measured_data_Flow.loc[T607_measured_data_Flow['distance_m'] < 605, 'flow'].mean()
print('measured avg flow upstream the fire in T607', T607_Flow_avg_upstream_mea)
T607_Flow_avg_downstream_mea = T607_measured_data_Flow.loc[T607_measured_data_Flow['distance_m'] > 605, 'flow'].mean()
print('measured avg flow downstream the fire in T607', T607_Flow_avg_downstream_mea)
T607_Temp_avg_upstream_mea = T607_measured_data_Temp.loc[T607_measured_data_Temp['distance_m'] < 605, 'Temp_C'].mean()
print('measured avg temperature upstream the fire in T607', T607_Temp_avg_upstream_mea)
T607_Temp_avg_downstream_mea = T607_measured_data_Temp.loc[T607_measured_data_Temp['distance_m'] > 605, 'Temp_C'].mean()
print('measured avg temperature upstream the fire in T607', T607_Temp_avg_downstream_mea)


plt.legend(fontsize=10)
plt.xlabel('Distance from North Portal (m)', fontsize=16)
plt.ylabel('Flowrate(m3/s)', fontsize=16)
plt.title('Test 607 - Flowrate Distribution', fontsize=12)
plt.tight_layout()
plt.show()



##############################################################################################

# Calculate average HRR as the mean between 'Based On Efficiency' and 'Raw (Calculated)' 
T607_HRR_data['HRR_avg'] = T607_HRR_data.iloc[:, [1, 2]].astype(float).mean(axis=1)

# Calculate negative flow from 'Loop 209'
avg_flow = -Q607_Flow['Loop 209'].astype(float)
time_flow = Q607_Flow['Time'].astype(float)
time_hrr = T607_HRR_data['Time'].astype(float)

# Find common time values between datasets
common_time = time_flow[time_flow.isin(time_hrr)]

# Match data based on common time values
avg_flow_common = avg_flow[time_flow.isin(common_time)].reset_index(drop=True)
hrr_common = T607_HRR_data[T607_HRR_data['Time'].isin(common_time)]['HRR_avg'].reset_index(drop=True)
common_time_values = common_time.reset_index(drop=True)

# Plot with dual y-axes
fig, ax1 = plt.subplots()

color = 'black'
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Flow (m³/s)', color=color)
line1, = ax1.plot(common_time_values.values, avg_flow_common.values, color=color, label='Upstream Flow')
ax1.set_xlim(0,1000)
ax1.set_xlim(left=0)  # Set x-axis to start at 0
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True)

ax2 = ax1.twinx()
color = 'gray'
ax2.set_ylabel('HRR (kW)', color=color)
line2, = ax2.plot(common_time_values.values, hrr_common.values, color=color, linestyle='--', label='HRR')
ax2.tick_params(axis='y', labelcolor=color)

# Add rectangle for the time range iloc[22:51]
start_time = common_time_values.iloc[28]
end_time = common_time_values.iloc[32]
ax1.axvspan(start_time, end_time, color='gray', alpha=0.3)

# Add legend below the plot close to x-axis
lines = [line1, line2]
labels = [line.get_label() for line in lines]
fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2)

plt.tight_layout()
plt.show()


###########################exporting results###########################################################
SES_results_14_MW = combined_results[combined_results['file'] == './SES_results/MT-T607-R3.xlsx']
SES_results_14_MW = SES_results_14_MW.drop('file', axis=1)
SES_results_14_MW.to_csv('./Outputs/SES_results_607_14MW.csv',index=False)

SES_results_20_MW = combined_results[combined_results['file'] == './SES_results/MT-T607-R4.xlsx']
SES_results_20_MW = SES_results_20_MW.drop('file', axis=1)
SES_results_20_MW.to_csv('./Outputs/SES_results_607_20MW.csv',index=False)

T607_measured_data_Flow.to_csv('./Outputs/T607_Q.csv',index=False)
T607_measured_data_Temp.to_csv('./Outputs/T607_T.csv',index=False)