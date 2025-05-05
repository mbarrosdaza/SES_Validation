import pandas as pd
import matplotlib.pyplot as plt
import glob
import itertools

#####################MEASURED HRR VALUES FROM FDS REPO - 'Raw (Calculated)' ########################

# Import the CSV file, skipping the second row (index 1)
df = pd.read_csv('./Measured_data/HRR606A.csv', skiprows=[1])

# Filter data for x-axis up to 2500
df_filtered = df[df['Time'] <= 2500]

# Calculate average HRR for both columns 1 and 2 in the filtered data
average_hrr_based_efficiency = df_filtered.iloc[22:51, 1].astype(float).mean()
average_hrr_raw_calculated = df_filtered.iloc[22:51, 2].astype(float).mean()
average_hrr = df_filtered.iloc[22:50, 1:3].astype(
    float).mean(axis=0).mean(axis=0) 

print("Average HRR (Based On Efficiency) from iloc[22:51]:", average_hrr_based_efficiency)
print("Average HRR (Raw Calculated) from iloc[22:51]:", average_hrr_raw_calculated)
print("Average HRR from iloc[22:51]:", average_hrr)

#####################MEASURED TEMPERATURE VALUES FROM FDS REPO + AVERAGING TEMP IN EACH LOOP ########################

loops = [202, 301, 302, 303, 304, 205, 305,
         306, 307, 207, 208, 209, 211, 213, 214]
loops_length_ft = [2736, 2373, 2236, 2116, 2059, 2019,
                   1982, 1923, 1816, 1668, 1399, 1053, 692, 347, 65]
loops_length_m = [ft * 0.3048 for ft in loops_length_ft]

T606A_Temp = pd.read_csv('./Measured_data/TP606A.csv')
SES_segment_length = pd.read_csv('./Measured_data/SES_segment_length.csv')
T606A_Temp = T606A_Temp.drop(index=0)
loop_n = []
T606A_data_Temp = []
a = 1
for loop in loops:
    loop_n.append(T606A_Temp.keys().str.contains(str(loop)).sum())
for n in loop_n:
    T606A_data_Temp.append(
        T606A_Temp.iloc[22:50, a:n+a].astype(float).mean(axis=1).mean(axis=0))
    a += n
T606A_measured_data_Temp = pd.DataFrame(
    {"loop": loops, "distance_m": loops_length_m, "Temp_C": T606A_data_Temp})
T606A_measured_data_Temp = T606A_measured_data_Temp.sort_values(by=['distance_m'])

plt.figure()
plt.plot(T606A_measured_data_Temp['distance_m'], T606A_measured_data_Temp['Temp_C'], marker='o', color='black')
plt.xlabel('Distance (m)')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.show()


##################### MEASURED FLOW VALUES FROM FDS REPO  ########################

loops = [214, 209, 208, 207, 307, 305, 304, 302, 301, 202]
loops_length_ft = [65, 1053, 1399, 1668, 1816, 1982, 2059, 2236, 2373, 2736]
loops_length_m = [ft * 0.3048 for ft in loops_length_ft]
Q606A_Flow = pd.read_csv('./Measured_data/QP606A.csv')
Q606A_Flow = Q606A_Flow.drop(index=0)
T606A_measured_Flow = abs(Q606A_Flow.iloc[22:50, 1:].astype(float).mean(axis=0)).tolist()
T606A_measured_Flow[0] = 98.087  # # Value corrected in Memorial Tunnel Phase 4 Report 1999 (Figure 7.6.3-1 Test 606A: Bulk flow and temperature (S5, S6, S7))
T606A_measured_Flow[9] = 122.2   # # Value corrected in Memorial Tunnel Phase 4 Report 1999 (Figure 7.6.3-1 Test 606A: Bulk flow and temperature (S5, S6, S7))
T606A_measured_data_Flow = pd.DataFrame(
    {"loop": loops, "distance_m": loops_length_m, "flow": T606A_measured_Flow})

plt.figure()
plt.plot(T606A_measured_data_Flow['distance_m'], T606A_measured_data_Flow['flow'], marker='o', color='black')
plt.xlabel('Distance (m)')
plt.ylabel('Flow (m³/s)')
plt.grid(True)
plt.show()


##################### SES RESULTS EXTRACTION  ###########################################################3

temp = []
file = './SES_results/MT-T606A-R3.xlsx'
data_t = pd.read_excel(file, sheet_name="Temperature")
data_q = pd.read_excel(file, sheet_name="Flow Rate")
temp = data_t.iloc[:, 65:106].mean(axis=1).tolist()
flow = data_q.iloc[:, 65:106].mean(axis=1).tolist()
pressure = 101325
R_constant = 287.05
density = [pressure/(R_constant*(273.15 + t)) for t in temp]
T606A_Temp = T606A_Temp.astype(float)
densi_upstream_606A = pressure/(R_constant*(273.15 + T606A_Temp.iloc[0,1:].mean()))
flow2 = [f * densi_upstream_606A/dens for f,
         dens in itertools.zip_longest(flow, density)]
SES_results_606A = pd.DataFrame({"length": SES_segment_length['distance'].tolist(
), "temperature": temp, "flowrate": flow2})

################## CORRECTING SES PREDICTED FLOW VALUES DUE TO ACTUAL DENSITY ##############################

pressure = 101325
R_constant = 287.05
density = [pressure/(R_constant*(273.15 + t)) for t in temp]
T606A_Temp = T606A_Temp.astype(float)
densi_upstream_606A = pressure/(R_constant*(273.15 + T606A_Temp.iloc[0,1:].mean()))
flow2 = [f * densi_upstream_606A/dens for f,
         dens in itertools.zip_longest(flow, density)]
SES_results_606A = pd.DataFrame({"length": SES_segment_length['distance'].tolist(
), "temperature": temp, "flowrate": flow2})


################ COMPARING/PLOTTING TEMPERATURE MEASURED VALUES VS SES PREDICTIONS ##############################

plt.plot(T606A_measured_data_Temp["distance_m"], T606A_measured_data_Temp["Temp_C"],
         label="Measured", marker="o", color='black', fillstyle='none')
plt.plot(SES_results_606A["length"], SES_results_606A["temperature"],
         label="SES", marker="s", color='blue', fillstyle='none')
plt.ylim(0, 125)
# plt.xticks([0, 200, 400, 600, 800, 1000])
plt.legend(fontsize=10)
plt.xlabel('Distance from North Portal (m)', fontsize=16)
plt.ylabel('Temperature(°C)', fontsize=16)
plt.title('Test 606A - Temperature Distribution', fontsize=12)
plt.tight_layout()
plt.show()

'upstream '
T606A_Temp_avg_upstream = SES_results_606A.loc[SES_results_606A['length'] < 605, 'temperature'].mean()
T606A_Temp_avg_upstream_mea = T606A_measured_data_Temp.loc[T606A_measured_data_Temp['distance_m'] < 605, 'Temp_C'].mean()
print('predicted avg temperature upstream the fire in T606A', T606A_Temp_avg_upstream)
print('measured avg temperature upstream the fire in T606A', T606A_Temp_avg_upstream_mea)

'Downstream '
T606A_Temp_avg_downstream = SES_results_606A.loc[SES_results_606A['length'] > 605, 'temperature'].mean()
T606A_Temp_avg_downstream_mea = T606A_measured_data_Temp[(T606A_measured_data_Temp['distance_m'] > 605) & (T606A_measured_data_Temp['Temp_C'] < 100)]['Temp_C'].mean()
print('predicted avg temperature downstream the fire in T606A', T606A_Temp_avg_downstream)
print('measured avg temperature downstream the fire in T606A', T606A_Temp_avg_downstream_mea)


################ COMPARING/PLOTTING FLOW MEASURED VALUES VS SES PREDICTIONS ##############################

plt.plot(T606A_measured_data_Flow["distance_m"], T606A_measured_data_Flow["flow"],
         label="Measured", marker="o", color='black', fillstyle='none')
plt.plot(SES_results_606A["length"], SES_results_606A["flowrate"],
         label="SES", marker="s", color='blue', fillstyle='none')
# plt.plot(T606A_measured_data_Flow["distance_m"], T606A_measured_data_Flow["flow_rho"],
#          label="Measured-adjusted", marker="s", color='red', fillstyle='none')
# plt.yticks([50, 75, 100, 125, 150, 175, 200])
plt.xticks([0, 200, 400, 600, 800, 1000])
plt.legend(fontsize=10)
plt.xlabel('Distance from North Portal (m)', fontsize=16)
plt.ylabel('Flowrate(m3/s)', fontsize=16)
plt.title('Test 606A - Flowrate Dsirtbution', fontsize=12)
plt.tight_layout()
plt.show()

'Upstream'

T606A_Flow_avg_upstream = SES_results_606A.loc[SES_results_606A['length'] < 605, 'flowrate'].mean()
T606A_Flow_avg_upstream_mea = T606A_measured_data_Flow.loc[T606A_measured_data_Flow['distance_m'] < 605, 'flow'].mean()
print('predicted avg flow upstream the fire in T606A', T606A_Flow_avg_upstream)
print('measured avg flow upstream the fire in T606A', T606A_Flow_avg_upstream_mea)

'Downstream '
T606A_Flow_avg_downstream = SES_results_606A.loc[SES_results_606A['length'] > 605, 'flowrate'].mean()
T606A_Flow_avg_downstream_mea = T606A_measured_data_Flow[(T606A_measured_data_Flow['distance_m'] > 605) & (T606A_measured_data_Flow['flow'] < 160)]['flow'].mean()
print('predicted avg Flow downstream the fire in T606A', T606A_Flow_avg_downstream)
print('measured avg Flow downstream the fire in T606A', T606A_Flow_avg_downstream_mea)


##############################################################################################

# Calculate average HRR as the mean between 'Based On Efficiency' and 'Raw (Calculated)' 
df_filtered['HRR_avg'] = df.iloc[:, [1, 2]].astype(float).mean(axis=1)

# Calculate negative flow from 'Loop 209'
avg_flow = -Q606A_Flow['Loop 209'].astype(float)
time_flow = Q606A_Flow['Time'].astype(float)
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
ax1.set_xlabel('Time')
ax1.set_ylabel('Flow (m³/s)', color=color)
line1, = ax1.plot(common_time_values.values, avg_flow_common.values, color=color, label='Upstream Flow')
ax1.set_ylim(-75, 175)
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
start_time = common_time_values.iloc[22]
end_time = common_time_values.iloc[50]
ax1.axvspan(start_time, end_time, color='gray', alpha=0.3)

# Add legend below the plot close to x-axis
lines = [line1, line2]
labels = [line.get_label() for line in lines]
fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2)


plt.tight_layout()
plt.show()




