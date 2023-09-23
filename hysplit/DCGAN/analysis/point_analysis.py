# This code performs the monitoring station analysis for the surrogate model.
# Plots are made, Jenson-Shannon Divergence is calculated and a KS test is performed.
# The names of the plots need to be adjusted according to your naming convention 

import os, sys, time
import pandas as pd
from scipy.stats import ks_2samp, pearsonr
from geopy import distance
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx
import geopandas as gpd
import torch
import seaborn as sns
import h5py
sns.set_style("whitegrid")
sns.set_context("talk", font_scale=2,rc={"lines.linewidth": 6})


columns = ['name', 'lat', 'lon', 'data_root', 'grid_id', 'num_down_geodf']
monitoring_stations = pd.DataFrame(columns=columns)
# MAY NEED TO ADJUST FILE PATHS
stations = [
    {'name':'Liberty', 'lat':40.323768, 'lon':-79.868062, 'data_root':'../../sensor_data/Liberty_emission_filled-1.csv', 'grid_id': (41,41), "num_down_geodf": 2641}, 
    {'name': 'Glassport', 'lat':40.326009, 'lon':-79.881703, 'data_root':'../../sensor_data/Glassport_emission_filled-1.csv', 'grid_id':(39,17), "num_down_geodf": 2513},
    {'name': 'Harrison', 'lat':40.617488, 'lon':-79.727664, 'data_root': '../../sensor_data/Harrison_emission_filled-1.csv', 'grid_id': None, "num_down_geodf": np.nan},
    {'name': 'NorthBraddock', 'lat':40.402324, 'lon':-79.860973, 'data_root':'../../sensor_data/NorthBraddock_emission_filled-1.csv', 'grid_id': (42,30), "num_down_geodf": 2718},
    {'name':'Lawrenceville', 'lat':40.465420, 'lon':-79.960757, 'data_root': '../../sensor_data/Lawrenceville_emission_filled-1.csv', 'grid_id': (29,57), "num_down_geodf": 1897}
]

monitoring_stations = pd.DataFrame(stations, columns=columns)
monitoring_stations_gdf = gpd.GeoDataFrame(monitoring_stations, geometry=gpd.points_from_xy(monitoring_stations['lon'], monitoring_stations['lat']))

def preprocess_monitor(df):
    df.index = pd.to_datetime(df.index)
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day
    df['Hour'] = df.index.hour
    return df

# find emission values at grid ids of sources
grid_ids = monitoring_stations['grid_id'].tolist()
# remove nans
grid_ids = [x for x in grid_ids if str(x) != 'None']
num_down_geodf = [x for x in monitoring_stations['num_down_geodf'].tolist() if str(x) != 'nan']
 
 
# LOADING DATA, ADJUST FILE PATHS, TEST NUM ETC
test_num = 22
input_info = torch.load(f'../results_images/test_{test_num}/state11579_full_input.pt')
dates = []
for i in range(len(input_info)):
    dates.append(input_info[i][-9])
dates = np.array(dates)
dates = dates * (1680300000 - 1470002400) + 1470002400
dates = pd.to_datetime(dates, unit='s')
dcgan_output = []

# with h5py.File(f'../results_images/test_{test_num}/tensors_state{iter_num}.h5', 'r') as hf:
with h5py.File(f'../results_images/test_{test_num}/tensors_state11579.h5', 'r') as hf:
    for dataset_name in hf:
        dcgan_output.append(hf[dataset_name][:])

# scale 
for i in range(len(dcgan_output)):
    grid = dcgan_output[i].squeeze()
    flat_grid = grid.flatten('F')
    #inverse min max scaling
    flat_grid_rescaled = (flat_grid - (-1.0)) * (35.0 - 0.0) / (1.0 - (-1.0)) + 0.0
    dcgan_output[i] = flat_grid_rescaled
    
# sort dcgan output by dates
sorted_tuples = sorted(zip(dates, dcgan_output))
date_dcgan_tuples = [(date, dcgan) for date, dcgan in sorted_tuples]



allresults = {}
for station in num_down_geodf:
    name = monitoring_stations_gdf[monitoring_stations_gdf['num_down_geodf'] == station]['name'].values[0]
    print(name)
    gdf = preprocess_monitor(pd.read_csv(monitoring_stations_gdf[monitoring_stations_gdf['name'] == name]['data_root'].values[0], index_col=0))
    predictions = []
    targets = []
    rmse_results = []
    for date, dcgan in date_dcgan_tuples:
        hour_prediction = dcgan[int(station)]
        predictions.append(hour_prediction)
        matching_date = gdf[(gdf['Year'] == date.year) & (gdf['Month'] == date.month) & (gdf['Day'] == date.day) & (gdf['Hour'] == date.hour)]
        target = matching_date['emission'].values[0]
        targets.append(target)
        if target > 0:
            rmse_calc = np.sqrt(np.mean((target - hour_prediction)**2))
            rmse_results.append(rmse_calc)
        else:
            rmse_results.append(np.nan)
        
    allresults[name] = {'predictions': predictions, 'targets': targets, 'rmse': rmse_results}


# extract the days from the sorted dates
dates = [x[0] for x in date_dcgan_tuples]
# sort dates into a list for each day
current_day = dates[0].date()
day_sublist = []
day_sublists = []
# Iterate through the sorted dates
for date in dates:
    # Check if the current date is different from the previous date
    if date.date() != current_day:
        day_sublists.append(day_sublist) 
        day_sublist = []  
        current_day = date.date()  
    
    day_sublist.append(date)  # Add the date to the current sublist

if day_sublist:
    day_sublists.append(day_sublist)

# find those indexes in the dates list where len of sublist is 1
for sublist in day_sublists:
    print(sublist) if len(sublist) == 1 else None

chunk_size = 24
fig, ax = plt.subplots(4,1,figsize=(20,20), sharey=True, sharex=True, layout = 'constrained')
plt.ylabel('Correlation Coefficient (R)')
plt.xticks(rotation=45)
plt.xlabel('Date')
all_station_corr = []
for num, station in enumerate(allresults.keys()):
    print(station)
    prediction_chunks = []
    targets_chunks = []
    date_chunks = []
    predictions = allresults[station]['predictions']
    print(len(predictions))
    date_list = dates
    print(len(date_list))
    targets = allresults[station]['targets']
    print(len(targets))
    
    # collect appropriate indexes to analyse
    indexes_to_remove = [2472, 6577, 16346]
    indexes_to_remove.sort(reverse=True)
    
    new_date_list = [date for index, date in enumerate(date_list) if index not in indexes_to_remove]
    new_predictions = [pred for index, pred in enumerate(predictions) if index not in indexes_to_remove]
    new_targets = [target for index, target in enumerate(targets) if index not in indexes_to_remove]
    
    print(len(new_date_list), len(new_predictions), len(new_targets))
    # change -1 in targets to nan
    new_targets = [np.nan if x==-1 else x for x in new_targets]
    
    for i in range(0, len(predictions), chunk_size):
        prediction_chunks.append(new_predictions[i:i+chunk_size])
        targets_chunks.append(new_targets[i:i+chunk_size])
        date_chunks.append(new_date_list[i:i+chunk_size])
        
    correlation_coeff = []
    for i in range(len(prediction_chunks)):
        # find the index of nans in the target chunks
        nans_idx = np.argwhere(np.isnan(targets_chunks[i]))
        # remove from target chunks
        targets_chunks[i] = np.delete(targets_chunks[i], nans_idx)
        # remove from predictions chunks
        prediction_chunks[i] = np.delete(prediction_chunks[i], nans_idx)
        if len(targets_chunks[i]) == 0:
            correlation_coeff.append(np.nan)
        else:    
            r,p = pearsonr(prediction_chunks[i], targets_chunks[i])
            correlation_coeff.append(r)
    
    all_station_corr.append(correlation_coeff)
    days_only = [x.date() for x in new_date_list if x.hour == 0]
    # morning/afternoon
    # days_only = [x.strftime('%Y-%m-%d %H:%M:%S') for x in new_date_list if x.hour == 0 or x.hour == 12]
    ax[num].scatter(days_only, correlation_coeff[:-1], alpha=0.5, color='black')
    ax[num].set_title(station)

plt.savefig('images/run22_r_all.pdf')

# plot distribution of r for each station
labels =['Liberty', 'Glassport', 'North Braddock', 'Lawrenceville']
fig,axis = plt.subplots(figsize=(15,10))
for i in range(len(all_station_corr)):
    sns.histplot(all_station_corr[i], kde=True, ax=axis, alpha=0.3,binwidth=0.05, binrange=(-1,1), label=labels[i])
plt.legend()    
plt.xlabel('Correlation Coefficient')
plt.tight_layout()
plt.savefig('images/run22_r_distribution.pdf')

# calc js divergence between each MS distributions and the ones from hysplit
from scipy.spatial import distance
from scipy.stats import ks_2samp
# read from txt file
hysplits = np.loadtxt('../../HYSPLIT_results/analysis test/plots/r_values.txt')
dcgans = all_station_corr

def make_probs(dcgan, bins=40):
 """ Input is results from a models during an analsyis method. Eg. the results from the path
 analysis for average aggregation of smell reports. These arrays come from the dictionaries collected.
 To get in the correct format, you convert the dictionary to a df, then use df[0].to_numpy() to get the array.
 """
 dcgan = dcgan[~np.isnan(dcgan)]
 dcgan_vals, dcgan_bins = np.histogram(dcgan, bins=bins, density=True)
 return dcgan_vals

for dcgan, hysplit, name in zip(dcgans, hysplits, labels):
 js_dist = distance.jensenshannon(make_probs(np.array(dcgan)), make_probs(hysplit))
 js_divergence = js_dist**2
 print(f'JS divergence: {js_divergence.round(5)} for {name}' )

# now for ksp test:
np.random.seed(42)
dcgan_nonan = []
for list in dcgans:
    print(len(list))
    print("number of nans", np.count_nonzero(np.isnan(list)))
    list = np.array(list)
    nonan = list[~np.isnan(list)]
    print(len(nonan))
    dcgan_nonan.append(nonan)
    
hysplits_nonan = [x[~np.isnan(x)] for x in hysplits]
for dcgan, hysplit, name in zip(dcgan_nonan, hysplits_nonan, labels):
    sampled_hysplit = np.random.choice(hysplit, size=len(dcgan), replace=False)
    print(len(sampled_hysplit))
    ks_stat, p_value = ks_2samp(dcgan, sampled_hysplit)
    print(f'KS test: {ks_stat}, {p_value} for {name}')

# Plotting the RMSE for each station, not a useful plot
sns.set_context("talk", font_scale=3,rc={"lines.linewidth": 1})
fig,axis = plt.subplots(4,1,figsize=(50,30), sharex=True, sharey=True)
plt.ylim(0, 200)
for i, station in enumerate(allresults.keys()):
    rmse = allresults[station]['rmse']
    axis[i].plot(dates, rmse, label=station)
    axis[i].set_title(station)
axis[3].set_xticks([pd.to_datetime('2018'), pd.to_datetime('2019'), pd.to_datetime('2020-01-01'), pd.to_datetime('2021-01-01'), pd.to_datetime('2022-01-01'), pd.to_datetime('2023-01-01')])
axis[3].set_xticklabels(['2018', '2019', '2020', '2021', '2022', '2023'])
plt.xlabel('Date')
axis[2].set_ylabel('RMSE')
plt.savefig('images/run22_rmse_monstation.pdf')
    
# plotting predictions and targets
sns.set_context("talk", font_scale=2.5)
fig,ax = plt.subplots(4,1,figsize=(40,20), sharex=True, layout = 'constrained')
for i, station in enumerate(allresults.keys()):
    predictions = allresults[station]['predictions'] 
    # mutliply by 10 to get back to ug/m3
    predictions = [x*10 for x in predictions]
    targets = allresults[station]['targets']
    ax[i].plot(dates, predictions, 'b', label='predictions', alpha=0.6)
    ax[i].plot(dates, targets, 'r', label='targets', alpha=0.5 )
    
    ax[i].set_title(station)
leg =plt.legend(['Observed value', 'Predicted value x10'])
for legobj in leg.legend_handles:
    legobj.set_linewidth(5.0)
    
ax[2].set_ylabel('PM2.5 Concentration ($\mu g/m^3$) ')
plt.savefig('images/run22_predictions_targets.pdf')
   
# PLOT PREDICTIONS, TARGETS AND RMSE FOR EACH STATION FOR A PARTICULAR DAY
date = '2022-09-16'
date_corr = []
first_date_ix = dates.index(pd.to_datetime(date))
full_date = dates[first_date_ix:first_date_ix+24]
# just the hours from the date
full_date = [x.hour for x in full_date]
fig, ax = plt.subplots(4,1,figsize=(20,20), sharey=True, sharex=True, layout = 'tight')
for num, station in enumerate(allresults.keys()): 
    predictions = allresults[station]['predictions'][first_date_ix:first_date_ix+24]
    targets = allresults[station]['targets'][first_date_ix:first_date_ix+24]
    rmse = allresults[station]['rmse'][first_date_ix:first_date_ix+24]
    corr = pearsonr(predictions, targets)
    ax[num].set_title(f'{station} with correlation {corr[0].round(4)}')
    ax[num].plot(full_date, predictions, label='predictions', alpha=0.5)
    ax[num].plot(full_date, targets, label='targets', alpha=0.5)
    ax[num].plot(full_date, rmse, label='rmse', alpha=0.5)
    # plot predicted times 100
    # predicted_times100 = [x*10 for x in predictions]
    # plt.plot(full_date, predicted_times100, label='predictions*10', alpha=0.5)
plt.legend(['Predictions', 'Observations', 'RMSE'])

# ADDING HYSPLIT TO PLOT
def convert_str_to_array(df):
    if type(df) == str:
        data_str = df.replace('[', '').replace(']', '')
    numpy_array = np.fromstring(data_str, sep=' ')
    return numpy_array

year, month, day = 2022, 3,17
date = f'{year}-{month}-{day}'
date_corr = []
hysplit = pd.read_csv(f'../../HYSPLIT_results/analysis test/results/negative_test/{year}_{month}_{day}_negativetest.csv', index_col=0)

first_date_ix = dates.index(pd.to_datetime(date))
full_date = dates[first_date_ix:first_date_ix+24]
full_date = [x.hour for x in full_date]
fig, ax = plt.subplots(4,1,figsize=(17,17), sharex=True)
plt.tight_layout()
for num, station in enumerate(allresults.keys()): 
    hysplit_pred = convert_str_to_array(hysplit.iloc[3][station])
    predictions = allresults[station]['predictions'][first_date_ix:first_date_ix+24]
    targets = allresults[station]['targets'][first_date_ix:first_date_ix+24]
    rmse = allresults[station]['rmse'][first_date_ix:first_date_ix+24]
    corr = pearsonr(predictions, targets)
    corr_hysplit = pearsonr(hysplit_pred, targets)
    ax[num].set_title(f'{station}, R: {corr[0].round(4)}, {corr_hysplit[0].round(4)}')
    ax[num].plot(full_date, predictions, label='predictions', alpha=0.5, color='green')
    ax[num].plot(full_date, hysplit_pred, label='hysplit predictions', alpha=0.5, color='blue')
    ax[num].plot(full_date, targets, label='targets', alpha=0.5, color='red', linestyle='--')
    # ax[num].plot(full_date, rmse, label='rmse', alpha=0.5)
    # plot predicted times 100
    # predicted_times100 = [x*10 for x in predictions]
    # plt.plot(full_date, predicted_times100, label='predictions*10', alpha=0.5)
    # plt.legend()
ax[2].legend(['DCGAN', 'HYSPLIT', 'Observations'], fontsize=35)
ax[2].set_ylabel('PM2.5 Concentration ($\mu g/m^3$) ')
plt.xlabel('Hour of day')

plt.savefig('images/monstation_withhysplit_20220317.pdf')

