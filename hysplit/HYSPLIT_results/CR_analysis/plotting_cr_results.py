# This code plots the validation results for the HYSPLIT data (very similar to the DCGAN version)
# Plots are:
# 1. The percentage agreement between the model and the observations at hourly intervals
# 2. The percentage agreement between the model and the observations at morning and afternoon intervals, after uncommenting and commenting out the appropriate lines
# 3. Histogram of the correlation coefficients (R) for the hourly and morning/afternoon data
# 4. An alternative histogram of the correlation coefficient (R) for the hourly and morning/afternoon data
# 5. The average smell value vs the average PM2.5 concentration for the period of interest (2022)
# 6. The sum smell value vs the average PM2.5 concentration for the period of interest (2022) when changing name of csv to load
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
import shapely
from shapely.geometry import Point
sns.set_style("whitegrid")
sns.set_context("talk", font_scale=1.75)
from tqdm import tqdm
import h5py
import torch
import matplotlib.colors as mcolors
import pickle
from io import StringIO


# MAY NEED TO CHANGE
directory = 'results/'  # Replace with the actual directory path
suffix1 = 'morningafternoon_results.csv'  # Replace with the desired prefix


# PlOT 1 (and 2)
files_in_directory = os.listdir(directory)
morning_afternoons = [filename for filename in files_in_directory if filename.endswith(suffix)]
hourly_files = os.listdir(directory+'hourly/')
pa_sum_plot = {}
pa_avg_plot = {}
corr_sum_plot = {}
corr_avg_plot = {}
fig, ax = plt.subplots(1,2,figsize=(20, 7),gridspec_kw={'width_ratios': [4, 1]}, sharey=True, layout='constrained')

hr_pa_sum_plot = {}
hr_pa_avg_plot = {}
hr_corr_sum_plot = {}
hr_corr_avg_plot = {}
for filename in hourly_files:
 data = pd.read_csv(directory+'hourly/'+filename, index_col='date')
 data.index = pd.to_datetime(data.index, format='%Y-%m-%d-%H')
 hourly=ax[0].scatter(data.index, data['percentage_agreement_avg'], label='hourly', alpha=0.3, color='red', marker='x')
 
 for i in range(len(data)):
  hr_pa_sum_plot[data.index[i]] = data['percentage_agreement_sum'][i]
  hr_pa_avg_plot[data.index[i]] = data['percentage_agreement_avg'][i]
  hr_corr_sum_plot[data.index[i]] = data['corr_sum'][i]
  hr_corr_avg_plot[data.index[i]] = data['corr_avg'][i]


for filename in morning_afternoons:
 data = pd.read_csv(directory+filename, index_col='date')
 data['category'] = [data.index[i].split('-')[-1] for i in range(len(data))]
 data['time'] = data['category'].apply(lambda x: 10 if x == 'morning' else 18)
 data['month'] = [data.index[i].split('-')[1] for i in range(len(data))]
 data['day'] = [data.index[i].split('-')[2] for i in range(len(data))]
 data['year'] = [data.index[i].split('-')[0] for i in range(len(data))]
 data['plot_date'] = pd.to_datetime(data['month'] + '-' + data['day'] + '-' + data['year']+ ' ' + data['time'].astype(str))
 data = data.drop(columns=['month', 'day', 'year', 'time'])
 # morning = ax[0].scatter(data['plot_date'], data['percentage agreement average'], label='average', alpha=0.3, color='blue')
 # morning =ax[0].scatter(data['plot_date'], data['percentage_agreement_avg'], label='morning afternoon', alpha=0.3, color='blue', marker='x')
 for i in range(len(data)):
  pa_sum_plot[data['plot_date'][i]] = data['percentage_agreement_sum'][i]
  pa_avg_plot[data['plot_date'][i]] = data['percentage_agreement_avg'][i]
  corr_sum_plot[data['plot_date'][i]] = data['corr_sum'][i]
  corr_avg_plot[data['plot_date'][i]] = data['corr_avg'][i]

# uncomment theses lines to plot the morning/afternoon data, also comment out the hourly data and change plot title
# hist_data = pd.DataFrame.from_dict(pa_avg_plot, orient='index')
# ax[1].hist(hist_data, orientation='horizontal', color='blue', alpha=0.3, label='Morning/Afternoon',  weights=np.ones_like(hist_data)*100 / len(hist_data))
hr_hist_data = pd.DataFrame.from_dict(hr_pa_avg_plot, orient='index')
ax[1].hist(hr_hist_data, orientation='horizontal', color='red', alpha=0.3, label='Hourly',  weights=np.ones_like(hr_hist_data)*100 / len(hr_hist_data))


ax[0].set_ylabel('% Agreement')
ax[1].set_xlabel('Frequency')
ax[0].set_ylim([-10, 110]) 
ax[0].set_xlabel('Date')
ax[0].set_xticks([pd.to_datetime('2018'), pd.to_datetime('2019'), pd.to_datetime('2020-01-01'), pd.to_datetime('2021-01-01'), pd.to_datetime('2022-01-01'), pd.to_datetime('2023-01-01')])
ax[0].set_xticklabels(['2018', '2019', '2020', '2021', '2022', '2023'])

plt.savefig('images/hysplit_pa_avg_hourly.pdf')
# plt.savefig('images/hysplit_pa_avg_MA.pdf')

# now for the impact
MA_corr_avg = pd.DataFrame.from_dict(corr_avg_plot, orient='index')
MA_corr_sum = pd.DataFrame.from_dict(corr_sum_plot, orient='index') 
#first the corrolations
corr_avg = pd.DataFrame.from_dict(hr_corr_avg_plot, orient='index')
nan_avg = np.count_nonzero(np.isnan(corr_avg))
corr_sum = pd.DataFrame.from_dict(hr_corr_sum_plot, orient='index')
nan_sum = np.count_nonzero(np.isnan(corr_sum))

# PLOT 3
arr_corr_avg = corr_avg[0].to_numpy()
arr_corr_sum = corr_sum[0].to_numpy()
arr_MA_corr_avg = MA_corr_avg[0].to_numpy()
arr_MA_corr_sum = MA_corr_sum[0].to_numpy()
all_corr = [arr_corr_avg, arr_corr_sum, arr_MA_corr_avg, arr_MA_corr_sum]
fig, ax = plt.subplots(figsize=(15, 10))
ax.hist(all_corr,bins=15, histtype='bar', alpha=0.7, label=['Hourly average', 'Hourly sum', 'Morning/Afternoon average', 'Morning/Afternoon sum'], density=False)
plt.xlabel('Correlation Coefficient (R)')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig('images/hysplit_corr_histogram.pdf')

# PLOT 4
fig, ax = plt.subplots(figsize=(15, 10))
labels = ['Hourly average', 'Hourly sum', 'Morning/Afternoon average', 'Morning/Afternoon sum']
for i, result in enumerate(all_corr):
 ax.hist(result,bins=15, histtype='step', alpha=0.7,weights=np.ones_like(result)*100/len(result), label=labels[i] , fill=False, linewidth=5 )
plt.xlabel('Correlation Coefficient (R)')
plt.ylabel('Frequency (%)')
plt.legend()
plt.savefig('images/hysplit_corr_histogram_alternative.pdf')


# PLOT 5 and 6
# load csv file for aggregation method of choice (average or sum)
combined_df_sum = pd.read_csv('combined_df_average_periodofimpact_hysplit.csv')
grouped = combined_df_sum.groupby('smell value')
fig, ax = plt.subplots(1,1, figsize=(10,10))
hists = []
means =[]
smells = []
binwidth = 2
for name, group in grouped:
    smells.append(name)
    means.append(group['TEST'].mean())
    ax.scatter(group['smell value'], group['TEST'], label=name, color='blue', alpha=0.5)
ax.scatter(smells, means, color='red', marker='x')
ax.plot(smells, means, 'black')
plt.xlabel('Average smell value')
plt.ylabel('PM2.5 concentration ($\mu g/m^3$)')
plt.tight_layout()
# save just in case
plt.savefig(f'images/hysplit_average_smellvalue_vs_emission_forperiodofinterest.pdf')
# %%
