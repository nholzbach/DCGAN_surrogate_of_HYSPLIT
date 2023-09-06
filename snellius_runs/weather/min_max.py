import xarray as xr
import os

file_path = "nc/"
datasets_list = []
for file_name in os.listdir(file_path):
    if file_name.endswith(".nc"):  # Assuming the files are in NetCDF format
        full_file_path = os.path.join(file_path, file_name)
        dataset = xr.open_dataset(full_file_path)
#        print(dataset)
        datasets_list.append(dataset)

# Step 2: Calculate the maximum value for each variable in each dataset
max_values_list = []
min_values_list = []

for dataset in datasets_list:
    max_values = dataset.max()# Calculate the maximum value for each variable
    min_values = dataset.min()
    max_values_list.append(max_values)
    min_values_list.append(min_values)

# Step 3: Find the maximum value across all datasets
combined_max_values = xr.concat(max_values_list, dim='data_vars').max(dim='data_vars')
combined_min_values = xr.concat(min_values_list, dim='data_vars').min(dim='data_vars')

print("max values: ", combined_max_values)
print("min values: ", combined_min_values)
