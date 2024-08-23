import os
import pickle
import sys
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
import data_preprocessor

data_path = '../data/original-data/continuous_factory_process.csv'
data_processor = data_preprocessor.DataProcessor(data_path)

data = data_processor.prepare_data_for_project(add_target=True)
print(data[data['Y_Target'] == 1])
data = data.drop(columns=['Y_Target'])

#print(data.loc[data['Y_Target'] == 1].mean())
#print(data.loc[data['Y_Target'] == 0].mean())

#data = data.filter(like='Actual')

#flattened_outputs = data.values.flatten()

# Calculate the overall variance
#overall_variance = np.var(flattened_outputs)

# Now you can compare the overall variance with the MSE
#print(f"Overall Variance: {overall_variance}")


#r_sqaured = 1 - (3.227299490383848 / overall_variance)

#print(data)

# Assuming 'data' is your first dataframe
# Calculate the mean and standard deviation of each column in 'data'
mean_of_cols = data.mean()
std_of_cols = data.std()

# Creating the dataframe with the given values
optimal_input_vals = [
    [12.86078, 2.1659892, 11.275402, 21.299019, 34.429867, 0.2804765,
     1.1240721, -0.28204966, 20.956661, 19.015516, 7.887615, -0.10663208,
     1.5038733, 3.2233784, 10.255644],
    [17.209448, 30.100248, 13.413141, 28.093538, 53.802105, 0.44157577,
     -0.2849056, -0.7491537, 31.932869, 28.425802, 11.521267, -1.8963693,
     3.62645, 3.790621, 18.64446],
     [12.931355,    8.589792,   11.247722,   21.357794,   33.785618,   0.16102208,
    0.7321482,  -0.06311998, 20.948696,   19.185188,    7.7999444,   0.04797417,
    1.5307819,   3.2587452,  6.675483],
    [13.034322,   11.863741,   11.921122,   23.861698,   34.444363,   -0.09679553,
    3.6107686,   3.7711446,  22.960777,   19.219057,    8.572758,    4.4011636,
    1.7976742,   1.9645234,  17.447441]
]
"""
# Use the same column names as in 'data'
columns = data.columns  # Assuming 'data' has the same column names as the new data

df2 = pd.DataFrame(optimal_input_vals, columns=columns)

# Create a new dataframe to store the result
result_df = pd.DataFrame(index=df2.index, columns=df2.columns)

# Iterate over each row of df2
for index, row in df2.iterrows():
    # Check if the value is within the range of mean +/- std for each column
    within_range = (row >= (mean_of_cols - 3 * std_of_cols)) & (row <= (mean_of_cols + 3 * std_of_cols))
    result_df.loc[index] = within_range
"""
#print(result_df)

#filename = 'optimization-results/optimization_results.pkl'
#with open(filename, 'rb') as f:
 #   results = pickle.load(f)

#print(results)


# Target values
target_values = np.array([13.75, 22.74, 13.02, 21.88, 32.55, 2.74, 4.25, 2.97, 21.3, 19.52, 8.65, 6.16, 2.02, 3.16, 17.72])

# Test arrays
nm_all_target = np.array([17.209448, 30.100248, 13.413141, 28.093538, 53.802105, 0.44157577,
     -0.2849056, -0.7491537, 31.932869, 28.425802, 11.521267, -1.8963693,
     3.62645, 3.790621, 18.64446])

bfgs_all = np.array([12.86078, 2.1659892, 11.275402, 21.299019, 34.429867, 
                         0.2804765, 1.1240721, -0.28204966, 20.956661, 19.015516, 
                         7.887615, -0.10663208, 1.5038733, 3.2233784, 10.255644])

bfgs_red = np.array([12.931355,    8.589792,   11.247722,   21.357794,   33.785618,   0.16102208,
  0.7321482,  -0.06311998, 20.948696,   19.185188,    7.7999444,   0.04797417,
  1.5307819,   3.2587452,  6.675483])

nm_red_target = np.array([13.034322,   11.863741,   11.921122,   23.861698,   34.444363,   -0.09679553,
  3.6107686,   3.7711446,  22.960777,   19.219057,    8.572758,    4.4011636,
  1.7976742,   1.9645234,  17.447441]) 


nm_red_mean = np.array([12.944496,   10.958598,   12.147692,   23.529,      33.580624,   -0.08617131,
  3.9634795,   3.8095765,  22.457003,   18.847347,    8.370346,    4.5729766,
  1.7465392,   1.8734632,  16.829742,
])

nm_all_mean = np.array([
    12.818185,    5.0574164,  11.152904,   21.40369,    34.074474,    0.13126153,
   0.74135697,  0.15738347, 20.886358,   18.95631,     7.769295,   0.17830679,
   1.5338256,   3.1157403,   6.071961, 
])


""" # Calculate mean absolute difference for each test array
mean_diff_1 = np.mean(np.abs(nm_all_target - target_values))
mean_diff_2 = np.mean(np.abs(bfgs_all - target_values))
mean_diff_3 = np.mean(np.abs(bfgs_red - target_values))
mean_diff_4 = np.mean(np.abs(nm_red_target - target_values))
mean_diff_5 = np.mean(np.abs(nm_red_mean - target_values))
mean_diff_6 = np.mean(np.abs(nm_all_mean - target_values))

# Output results
print(f"Mean absolute difference for Test Array 1: {mean_diff_1}")
print(f"Mean absolute difference for Test Array 2: {mean_diff_2}")
print(f"Mean absolute difference for Test Array 2: {mean_diff_3}")
print(f"Mean absolute difference for Test Array 2: {mean_diff_4}")
print(f"Mean absolute difference for Test Array 2: {mean_diff_5}")
print(f"Mean absolute difference for Test Array 2: {mean_diff_6}")

filename = 'result-charts-3stds/comparison-measurement-all.pdf'
labels = ['All BFGS, CG, TR', 'All NM\nbased on target', 'All NM\nbased on mean', 'Reduced BFGS,CG, TR', 'Reduced NM\nbased on target', 'Reduced NM\nbased on mean']
mse_values = [mean_diff_2, mean_diff_1, mean_diff_6, mean_diff_3, mean_diff_4, mean_diff_5]

plt.figure(figsize=(10, 6))

# Define the x-coordinates for the bars
x_coords = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]  # Adjusted positions to space the bars evenly

# Plot the bars with the adjusted x-coordinates
plt.bar(x_coords, mse_values, color=['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#ff9f40'], width=0.15)  # Adjust width as needed
plt.xticks(x_coords, labels)  # Set the x-ticks to the labels at the correct positions

plt.xlabel('Output according to Input suggested by Algorithms')
plt.ylabel('Mean Absolute Difference')

# Set the y-axis limits to make small deviations more noticeable
#plt.ylim(2.5, 3.5)

# Save and show the plot
plt.savefig(filename, format='pdf')  
plt.show() """







def plot_roc_points(models):
    """
    Plots single points on a ROC space for multiple models.

    Parameters:
        models (dict): A dictionary where the keys are model names and the values are tuples of (TP, FP, TN, FN).
    """
    plt.figure(figsize=(8, 8))
    
    for model_name, (TP, FP, TN, FN) in models.items():
        TPR = TP / (TP + FN)  # True Positive Rate (Sensitivity, Recall)
        FPR = FP / (FP + TN)  # False Positive Rate (1 - Specificity)
        
        plt.scatter(FPR, TPR, label=model_name)
        plt.text(FPR + 0.01, TPR, model_name, fontsize=12)  # Add model name next to the point
    
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random classifier
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Space: Model Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()


model = {
    'SAE_1' : {274, 49, 236, 87},
    'DAE_1' : {274, 49, 235, 88},
    'DenAE_1' : {274, 49, 235, 88},
    'CAE_1' : {263, 60, 123, 200},
    'VAE_1' : {296, 27, 141, 182},
    'SAE_3' : {434, 1, 431, 4},
    'DAE_3' : {433, 2, 430, 5},
    'DenAE_3' : {433, 2, 430, 5},
    'CAE_3' : {419, 16, 410, 25},
    'VAE_3' : {423, 12, 417, 18}
}

#plot_roc_points(model)

cg = [
12.04, 216.77, 1044.59, 250.86, 1257.66, 72.02, 72.04, 75.12, 11.46,
423.54, 81.50, 77.99, 12.88, 240.35, 569.72, 256.29, 207.24, 69.03,
69.15, 73.51, 13.91, 226.54, 77.06, 60.06, 9.27, 218.53, 426.95, 204.17,
211.69, 78.01, 78.08, 346.65, 13.30, 252.56, 76.13, 65.06, 110.67, 86.75,
80.08
]

nm_mean = [
    12.75, 226.44, 1411.36, 253.21, 1256.54, 72.03, 72.00, 117.89, 10.38,
    384.32, 81.55, 82.68, 13.34, 241.05, 582.77, 256.06, 192.23, 68.51,
    69.18, 71.56, 14.07, 226.59, 76.43, 59.96, 8.93, 61.46, 437.30, 224.50,
    275.57, 78.01, 78.84, 352.17, 13.37, 259.34, 77.07, 65.54, 114.46, 95.72,
    76.77
]

nm_target = [
    11.05, 225.15, 1257.08, 252.12, 1235.34, 72.04, 72.02, 78.83, 6.19, 387.71,
    81.52, 90.40, 12.97, 240.12, 564.64, 256.49, 224.21, 68.70, 69.31, 72.73,
    14.02, 228.80, 76.66, 60.08, 9.25, 197.89, 435.10, 234.45, 226.59, 78.02,
    78.46, 337.39, 13.37, 230.22, 91.60, 65.61, 114.01, 91.66, 76.76
]
cg_red = [211.82675162, 205.11679483, 432.04233628,  12.85755822,
        12.29048741, 256.28261115,  86.01178237,  12.29048741,
        77.2251003 ,  11.52654865,   9.09786711,  13.68952778,
       251.84996994, 412.78172171, 974.51470229, 412.78172171,
       209.23886927,  81.96439341,  75.03276882, 571.7312197 ,
        72.26595251,  71.11619933, 347.75706827, 240.66564312,
       111.52706009,  72.06236334
]

nm_target_red = [191.62559332,  206.68943402,  476.59395878,   12.5452423 ,
    12.73169732,  256.72004436,   84.17434397,   12.73169732,
    78.53119051,    9.49940785,    9.13953357,   11.19349111,
229.37500141,  448.14752719, 1653.93955834,  448.14752719,
144.97626338,   80.60452397,   71.79392406,  557.21519192,
    66.40790009,   67.97842705,  335.02193096,  240.87383913,
117.54230177,   73.57662664]

nm_mean_red = [190.56736817,  207.09046449,  480.86311334,   12.55022379,   12.8483538,
    256.74028335,   84.60005638,   12.8483538,    78.00704708,    9.43263204,
    9.14521709,   11.55275842,  233.255311,    449.81821303, 1624.40206523,
    449.81821303,  142.62898312,   80.7767634,    71.24857486,  557.18994531,
    66.00161476,   67.7800658,   334.27307085,  240.69350509,  117.74184572,
    73.58217034

]

feature_names = [    'Machine1RawMaterialProperty1', 'Machine1RawMaterialProperty2', 'Machine1RawMaterialProperty3', 
    'Machine1RawMaterialProperty4', 'Machine1RawMaterialFeederParameter', 'Machine1Zone1Temperature', 
    'Machine1Zone2Temperature', 'Machine1MotorAmperage', 'Machine1MotorRPM', 'Machine1MaterialPressure', 
    'Machine1MaterialTemperature', 'Machine1ExitZoneTemperature', 'Machine2RawMaterialProperty1', 
    'Machine2RawMaterialProperty2', 'Machine2RawMaterialProperty3', 'Machine2RawMaterialProperty4', 
    'Machine2RawMaterialFeederParameter', 'Machine2Zone1Temperature', 'Machine2Zone2Temperature', 
    'Machine2MotorAmperage', 'Machine2MotorRPM', 'Machine2MaterialPressure', 'Machine2MaterialTemperature', 
    'Machine2ExitZoneTemperature', 'Machine3RawMaterialProperty1', 'Machine3RawMaterialProperty2', 
    'Machine3RawMaterialProperty3', 'Machine3RawMaterialProperty4', 'Machine3RawMaterialFeederParameter', 
    'Machine3Zone1Temperature', 'Machine3Zone2Temperature', 'Machine3MotorAmperage', 'Machine3MotorRPM', 
    'Machine3MaterialPressure', 'Machine3MaterialTemperature', 'Machine3ExitZoneTemperature', 
    'FirstStageombinerOperationTemperature1', 'FirstStageombinerOperationTemperature2', 
    'FirstStageombinerOperationTemperature3']

top_30_features = [
        "Machine3RawMaterialProperty2",
        "Machine3RawMaterialProperty4",
        "Machine3RawMaterialProperty3",
        "Machine2RawMaterialProperty1",
        "Machine1RawMaterialProperty1",
        "Machine2RawMaterialProperty4",
        "FirstStageombinerOperationTemperature2",
        "Machine1RawMaterialProperty1",
        "Machine1ExitZoneTemperature",
        "Machine1MotorRPM",
        "Machine3RawMaterialProperty1",
        "Machine3MotorRPM",
        "Machine3MaterialPressure",
        "Machine1MaterialPressure",
        "Machine1RawMaterialProperty3",
        "Machine1MaterialPressure",
        "Machine1RawMaterialProperty2",
        "Machine1MaterialTemperature",
        "Machine3MaterialTemperature",
        "Machine2RawMaterialProperty3",
        "Machine1Zone2Temperature",
        "Machine1MotorAmperage",
        "Machine3MotorAmperage",
        "Machine2RawMaterialProperty2",
        "FirstStageombinerOperationTemperature1",
        "Machine1Zone1Temperature"
]
# Create a DataFrame from your lists
df_values = pd.DataFrame({
    'All BFGS/CG/TR': cg,
    'All NM based on mean': nm_mean,
    'All NM based on target': nm_target
}, index=[
    'Machine1RawMaterialProperty1', 'Machine1RawMaterialProperty2', 'Machine1RawMaterialProperty3', 
    'Machine1RawMaterialProperty4', 'Machine1RawMaterialFeederParameter', 'Machine1Zone1Temperature', 
    'Machine1Zone2Temperature', 'Machine1MotorAmperage', 'Machine1MotorRPM', 'Machine1MaterialPressure', 
    'Machine1MaterialTemperature', 'Machine1ExitZoneTemperature', 'Machine2RawMaterialProperty1', 
    'Machine2RawMaterialProperty2', 'Machine2RawMaterialProperty3', 'Machine2RawMaterialProperty4', 
    'Machine2RawMaterialFeederParameter', 'Machine2Zone1Temperature', 'Machine2Zone2Temperature', 
    'Machine2MotorAmperage', 'Machine2MotorRPM', 'Machine2MaterialPressure', 'Machine2MaterialTemperature', 
    'Machine2ExitZoneTemperature', 'Machine3RawMaterialProperty1', 'Machine3RawMaterialProperty2', 
    'Machine3RawMaterialProperty3', 'Machine3RawMaterialProperty4', 'Machine3RawMaterialFeederParameter', 
    'Machine3Zone1Temperature', 'Machine3Zone2Temperature', 'Machine3MotorAmperage', 'Machine3MotorRPM', 
    'Machine3MaterialPressure', 'Machine3MaterialTemperature', 'Machine3ExitZoneTemperature', 
    'FirstStageombinerOperationTemperature1', 'FirstStageombinerOperationTemperature2', 
    'FirstStageombinerOperationTemperature3'
])

df_mean = pd.DataFrame({
    'Mean': mean_of_cols  # 39 values
}, index=df_values.index)

df_std = pd.DataFrame({
    'Std': std_of_cols  # 39 values
}, index=df_values.index)

new_features_df = pd.DataFrame({
    'Reduced BFGS/CG/TR': cg_red,
    'Reduced NM based on mean': nm_mean_red,
    'Reduced NM based on target': nm_target_red
}, index=top_30_features).round(2)

combined_df = df_values.join(new_features_df)
combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
combined_df = combined_df.reindex(df_values.index)
"""combined_df.insert(0, 'Feature', feature_names)
base_filename = 'result-charts-3stds/all_optimal_input_features_values_df.pdf'

fig, ax = plt.subplots(figsize=(12, len(feature_names) * 0.5))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=combined_df.values, colLabels=combined_df.columns, cellLoc='center', loc='center')

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)
table.auto_set_column_width(col=list(range(len(combined_df.columns))))

for (row, col), cell in table.get_celld().items():
    if row == 0:  # Header row
        cell.set_text_props(fontproperties=FontProperties(weight='bold'))

plt.tight_layout()
plt.savefig(base_filename, format='pdf')
plt.show()
plt.close(fig) 
print(combined_df)"""

# Define the range within which the values should fall (e.g., ±10% of the mean)
tolerance = 0.1  # 10% tolerance
# Calculate lower and upper bounds based on the mean
lower_bound = df_mean['Mean'] - (3 * df_std['Std'])
upper_bound = df_mean['Mean'] + (3 * df_std['Std'])
print(lower_bound)

# Initialize the DataFrame to store the results
within_range_df = pd.DataFrame(index=combined_df.index, columns=combined_df.columns)

# Apply the range check for each column
for column in combined_df.columns:
    within_range_df[column] = np.where(
        combined_df[column].isna(),
        np.nan,  # Preserve NaN values
        (combined_df[column] >= lower_bound) & (combined_df[column] <= upper_bound)
    )

""" 
within_range_df = within_range_df.applymap(lambda x: x if pd.isna(x) else bool(x))
within_range_df = within_range_df.fillna('--')
within_range_df.insert(0, 'Feature', feature_names)
# Print or inspect the resulting DataFrame
print(within_range_df)

base_filename = 'result-charts-3stds/within_range_df.pdf'

fig, ax = plt.subplots(figsize=(18, len(feature_names) * 0.5))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=within_range_df.values, colLabels=within_range_df.columns, cellLoc='center', loc='center')

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)
table.auto_set_column_width(col=list(range(len(within_range_df.columns))))

for (row, col), cell in table.get_celld().items():
    if row == 0:  # Header row
        cell.set_text_props(fontproperties=FontProperties(weight='bold'))

plt.tight_layout()
plt.savefig(base_filename, format='pdf')
plt.show()
plt.close(fig)  """

