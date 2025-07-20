# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 06:03:47 2025

@author: mdaza
"""

import pandas as pd
import matplotlib.pyplot as plt
import glob
import itertools

#####################MEASURED HRR VALUES  ########################

# Import the CSV file, skipping the second row (index 1)
df = pd.read_csv('./Measured_data/HRR611.csv', skiprows=[1])

# Filter data for x-axis up to 2500
df_filtered = df[df['Time'] <= 2500]

# Calculate average HRR for both columns 1 and 2 in the filtered data
# average_hrr_based_efficiency = df_filtered.iloc[22:51, 1].astype(float).mean()
# average_hrr_raw_calculated = df_filtered.iloc[22:51, 2].astype(float).mean()
subset = df.iloc[12:30, 1:3].astype(float)
average_hrr = subset.mean().mean()
min_value = subset.min().mean()
max_value = subset.max().mean()

# print("Average HRR (Based On Efficiency) from iloc[22:51]:", average_hrr_based_efficiency)
# print("Average HRR (Raw Calculated) from iloc[22:51]:", average_hrr_raw_calculated)
print("Minimum HRR from iloc[12:30]:", min_value)
print("Average HRR from iloc[12:30]:", average_hrr)
print("Maximum HRR from iloc[12:30]:", max_value)



#####################MEASURED TEMPERATURE VALUES  ########################

loops = [202, 301, 302, 303, 304, 205, 305,
         306, 307, 207, 208, 209, 211, 213, 214]
loops_length_ft = [2736, 2373, 2236, 2116, 2059, 2019,
                   1982, 1923, 1816, 1668, 1399, 1053, 692, 347, 65]
loops_length_m = [ft * 0.3048 for ft in loops_length_ft]

T611_Temp = pd.read_csv('./Measured_data/TP611.csv')
SES_segment_length = pd.read_csv('./Measured_data/SES_segment_length.csv')
T611_Temp = T611_Temp.drop(index=0)
loop_n = []
T611_data_Temp = []
a = 1
for loop in loops:
    loop_n.append(T611_Temp.keys().str.contains(str(loop)).sum())
for n in loop_n:
    T611_data_Temp.append(
        T611_Temp.iloc[12:30, a:n+a].astype(float).mean(axis=1).mean(axis=0))
    a += n
T611_measured_data_Temp = pd.DataFrame(
    {"loop": loops, "distance_m": loops_length_m, "Temp_C": T611_data_Temp})
T611_measured_data_Temp = T611_measured_data_Temp.sort_values(by=['distance_m'])

plt.figure()
plt.plot(T611_measured_data_Temp['distance_m'], T611_measured_data_Temp['Temp_C'], marker='o', color='black')
plt.xlabel('Distance (m)')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.show()


##################### MEASURED FLOW VALUES  ########################

loops = [214, 209, 208, 207, 307, 305, 304, 302, 301, 202]
loops_length_ft = [65, 1053, 1399, 1668, 1816, 1982, 2059, 2236, 2373, 2736]
loops_length_m = [ft * 0.3048 for ft in loops_length_ft]
Q611_Flow = pd.read_csv('./Measured_data/QP611.csv')
Q611_Flow = Q611_Flow.drop(index=0)
T611_measured_Flow = abs(Q611_Flow.iloc[12:30, 1:].astype(float).mean(axis=0)).tolist()

T611_measured_data_Flow = pd.DataFrame(
    {"loop": loops, "distance_m": loops_length_m, "flow": T611_measured_Flow})

plt.figure()
plt.plot(T611_measured_data_Flow['distance_m'], T611_measured_data_Flow['flow'], marker='o', color='black')
plt.xlabel('Distance (m)')
plt.ylabel('Flow (m³/s)')
plt.grid(True)
plt.show()


##################### SES RESULTS EXTRACTION  ###########################################################3

import matplotlib.pyplot as plt
import pandas as pd
import itertools

# SES results extraction with HRR sensitivity analysis
files = [
    './SES_results/MT-T611-R3-1.xlsx',   # Min HRR
    './SES_results/MT-T611-R3-2.xlsx',   # Min HRR (different variation)
    './SES_results/MT-T611-R3.xlsx',     # Average HRR
    './SES_results/MT-T611-R3-3.xlsx',   # Average HRR with variation
    './SES_results/MT-T611-R3-4.xlsx',   # Max HRR
    './SES_results/MT-T611-R3-5.xlsx'    # Max HRR (different variation)
]

labels = ['MIN-75%', 'MIN-83%', 'AVE-75%', 'AVE-83%', 'MAX-75%', 'MAX-83%' ]

pressure = 101325
R_constant = 287.05
ses_results_all = []

# Extract SES results
for file in files:
    data_t = pd.read_excel(file, sheet_name="Temperature")
    data_q = pd.read_excel(file, sheet_name="Flow Rate")
    temp = data_t.iloc[:, 65:106].mean(axis=1).tolist()
    flow = data_q.iloc[:, 65:106].mean(axis=1).tolist()
    density = [pressure / (R_constant * (273.15 + t)) for t in temp]
    T611_Temp = T611_Temp.astype(float)
    densi_upstream_611 = pressure / (R_constant * (273.15 + T611_Temp.iloc[0, 1:].mean()))
    flow2 = [f * densi_upstream_611 / dens for f, dens in itertools.zip_longest(flow, density)]
    ses_results_all.append(pd.DataFrame({
        "length": SES_segment_length['distance'].tolist(),
        "temperature": temp,
        "flowrate": flow2
    }))

# Plotting Temperature Distribution
plt.figure()
plt.plot(T611_measured_data_Temp["distance_m"], T611_measured_data_Temp["Temp_C"],
         label="Measured", marker="o", color='black', fillstyle='none')

# Plot each SES result curve with distinct labels and markers
markers = ['^', 'D', 's', 'v', 'p', 'H']  # Assigning different markers to each curve

plt.axvline(x=615, color='black', linestyle='--', label='Fire Location', alpha=0.7, linewidth=1)


plt.text(0.3, 0.5, 'Upstream', fontsize=8, transform=plt.gca().transAxes, 
         verticalalignment='top', horizontalalignment='left', color='black')
plt.text(0.93, 0.95, 'Downstream', fontsize=8, transform=plt.gca().transAxes, 
         verticalalignment='top', horizontalalignment='right', color='black')

a = 0
for i, df in enumerate(ses_results_all):
    # Highlight the average HRR result with a distinct style (blue with square marker)
    if i == 2:  # Average HRR (file index 2)
        plt.plot(df["length"], df["temperature"], label=labels[a], color='blue', fillstyle='none', marker=markers[i])
        T611_Temp_avg_upstream = df.loc[df['length'] < 605, 'temperature'].mean()
        print('predicted avg temperature upstream the fire in T611' + labels[a], T611_Temp_avg_upstream)
        T611_Temp_avg_downstream = df.loc[df['length'] > 605, 'temperature'].mean()
        print('predicted avg temperature downstream the fire in T611' + labels[a], T611_Temp_avg_downstream)
    # Highlight the average HRR with variation result with a distinct style (green with square marker)
    elif i == 3:  # Average HRR with variation (file index 3)
        plt.plot(df["length"], df["temperature"], label=labels[a], color='green', fillstyle='none', marker=markers[i])
        T611_Temp_avg_upstream = df.loc[df['length'] < 605, 'temperature'].mean()
        print('predicted avg temperature upstream the fire in T611' + labels[a], T611_Temp_avg_upstream)
        T611_Temp_avg_downstream = df.loc[df['length'] > 605, 'temperature'].mean()
        print('predicted avg temperature downstream the fire in T611' + labels[a], T611_Temp_avg_downstream)
    # Plot the other curves in grey scale, dashed, and slightly faded with markers
    else:
        plt.plot(df["length"], df["temperature"], label=labels[a], color='grey', fillstyle='none', alpha=0.7, marker=markers[i])
        T611_Temp_avg_upstream = df.loc[df['length'] < 605, 'temperature'].mean()
        print('predicted avg temperature upstream the fire in T611' + labels[a], T611_Temp_avg_upstream)
        T611_Temp_avg_downstream = df.loc[df['length'] > 605, 'temperature'].mean()
        print('predicted avg temperature downstream the fire in T611' + labels[a], T611_Temp_avg_downstream)
    a += 1


plt.legend(fontsize=8, ncol=2, loc='upper left')
plt.xlabel('Distance from North Portal (m)', fontsize=16)
plt.ylabel('Temperature (°C)', fontsize=16)
plt.title('Test 611 - Temperature Distribution', fontsize=12)
plt.tight_layout()

# Save the plot with high resolution (DPI = 300)
plt.savefig('Test_611_Temperature_Distribution.png', dpi=300)

plt.show()

# Plotting Flowrate Distribution
plt.figure()
plt.plot(T611_measured_data_Flow["distance_m"], T611_measured_data_Flow["flow"],
         label="Measured", marker="o", color='black', fillstyle='none')

# Plot each SES result curve with distinct labels and markers

plt.axvline(x=615, color='black', linestyle='--', label='Fire Location', alpha=0.7, linewidth=1)


plt.text(0.3, 0.5, 'Upstream', fontsize=8, transform=plt.gca().transAxes, 
         verticalalignment='top', horizontalalignment='left', color='black')
plt.text(0.95, 0.95, 'Downstream', fontsize=8, transform=plt.gca().transAxes, 
         verticalalignment='top', horizontalalignment='right', color='black')

a = 0
for i, df in enumerate(ses_results_all):
    # Highlight the average HRR result with a distinct style (blue with square marker)
    if i == 2:  # Average HRR (file index 2)
        plt.plot(df["length"], df["flowrate"], label=labels[a], color='blue', fillstyle='none', marker=markers[i])
        T611_Flow_avg_upstream = df.loc[df['length'] < 605, 'flowrate'].mean()
        print('predicted avg flow upstream the fire in T611' + labels[a], T611_Flow_avg_upstream)
        T611_Flow_avg_downstream = df.loc[df['length'] > 605, 'flowrate'].mean()
        print('predicted avg Flow downstream the fire in T611' + labels[a], T611_Flow_avg_downstream)
    # Highlight the average HRR with variation result with a distinct style (green with square marker)
    elif i == 3:  # Average HRR with variation (file index 3)
        plt.plot(df["length"], df["flowrate"], label=labels[a], color='green', fillstyle='none', marker=markers[i])
        T611_Flow_avg_upstream = df.loc[df['length'] < 605, 'flowrate'].mean()
        print('predicted avg flow upstream the fire in T611' + labels[a], T611_Flow_avg_upstream)
        T611_Flow_avg_downstream = df.loc[df['length'] > 605, 'flowrate'].mean()
        print('predicted avg Flow downstream the fire in T611' + labels[a], T611_Flow_avg_downstream)
    # Plot the other curves in grey scale, dashed, and slightly faded with markers
    else:
        plt.plot(df["length"], df["flowrate"], label=labels[a], color='grey', fillstyle='none', alpha=0.7, marker=markers[i])
        T611_Flow_avg_upstream = df.loc[df['length'] < 605, 'flowrate'].mean()
        print('predicted avg flow upstream the fire in T611' + labels[a], T611_Flow_avg_upstream)
        T611_Flow_avg_downstream = df.loc[df['length'] > 605, 'flowrate'].mean()
        print('predicted avg Flow downstream the fire in T611' + labels[a], T611_Flow_avg_downstream)
    a += 1
    
# plt.ylim(75,200)
plt.legend(fontsize=8, ncol=2, loc='upper left')
plt.xlabel('Distance from North Portal (m)', fontsize=16)
plt.ylabel('Flowrate (m³/s)', fontsize=16)
plt.title('Test 611 - Flowrate Distribution', fontsize=12)
plt.tight_layout()

# Save the plot with high resolution (DPI = 300)
plt.savefig('Test_611_Flowrate_Distribution.png', dpi=300)

plt.show()

##############################################################################################

# Calculate average HRR as the mean between 'Based On Efficiency' and 'Raw (Calculated)' 
df = pd.read_csv('./Measured_data/HRR611.csv', skiprows=[1])
df_filtered['HRR_avg'] = df.iloc[:, [1, 2]].astype(float).mean(axis=1)

# Calculate negative flow from 'Loop 209'
avg_flow = -Q611_Flow['Loop 209'].astype(float)
time_flow = Q611_Flow['Time'].astype(float)
time_hrr = df_filtered['Time'].astype(float)

# Find common time values between datasets
common_time = time_flow[time_flow.isin(time_hrr)]

# Match data based on common time values
avg_flow_common = avg_flow[time_flow.isin(common_time)].reset_index(drop=True)
hrr_common = df_filtered[df_filtered['Time'].isin(common_time)]['HRR_avg'].reset_index(drop=True)
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
# ax1.title('Test 606A - HRR & Flowrate', fontsize=12)
ax1.grid(True)

ax2 = ax1.twinx()
color = 'gray'
ax2.set_ylabel('HRR (kW)', color=color)
line2, = ax2.plot(common_time_values.values, hrr_common.values, color=color, linestyle='--', label='HRR')
ax2.tick_params(axis='y', labelcolor=color)

# Add rectangle for the time range iloc[22:51]
start_time = common_time_values.iloc[12]
end_time = common_time_values.iloc[30]
ax1.axvspan(start_time, end_time, color='gray', alpha=0.3)

# Add legend below the plot close to x-axis
lines = [line1, line2]
labels = [line.get_label() for line in lines]
fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2)


plt.tight_layout()

plt.savefig('Test_611_HRR.png', dpi=300)

plt.show()


###########################exporting results###########################################################
import os

tab_names = ['MT-T611-R3-1.xlsx','MT-T611-R3-2.xlsx', 'MT-T611-R3.xlsx', 'MT-T611-R3-3.xlsx', 'MT-T611-R3-4.xlsx', 'MT-T611-R3-5.xlsx' ]

with pd.ExcelWriter('Outputs/SES_results_611.xlsx', engine='openpyxl') as writer:
    for i, df in enumerate(ses_results_all,0):
        sheet_name = tab_names[i]
        df.to_excel(writer, sheet_name=sheet_name, index=False)
   
T611_measured_data_Flow.to_csv('./Outputs/T611_Q.csv',index=False)
T611_measured_data_Temp.to_csv('./Outputs/T611_T.csv',index=False)