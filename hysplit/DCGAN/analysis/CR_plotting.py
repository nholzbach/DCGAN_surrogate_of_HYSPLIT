
# This file plots the results from the CR analysis. 
# There are 7 plots:
# 1. The percentage agreement between the models and the citizen reports at an hourly level
# 2. The percentage agreement between the models and the citizen reports at a morning/afternoon level
# 3. The correlation coefficient between the models and the citizen reports at an hourly level (with both aggregation methods: sum and average)
# 4. The correlation coefficient between the models and the citizen reports at a morning/afternoon level and hourly level for the SUM aggregation method
# 5. The correlation coefficient between the models and the citizen reports at a morning/afternoon level and hourly level for the AVERAGE aggregation method
# 6. The DISTRIBUTION of the correlation coefficients for the models and the citizen reports at both time resolutions and aggregation methods
# 7. An alternative drawing of this distribution, using the step method

# After the plots, the Jenson-Shannon Divergence is calculated for two scenarios:
# 1. The JSD between all the evaluation metrics for the DCGAN and HYSPLIT models (IE comparing models)
# 2. The JSD between the different aggregation methods for each evaluation metrics for both the DCGAN and HYSPLIT models (IE comparing aggregation methods)

import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
import shapely
from shapely.geometry import Point
sns.set_style("whitegrid")
sns.set_context("talk", font_scale=2)
from tqdm import tqdm
import h5py
import torch
import matplotlib.colors as mcolors
import pickle
from io import StringIO
from scipy.spatial import distance


directory = 'stats/'  # Replace with the actual directory path
prefix = 'run22_11579_morningafternoon'  # Replace with the desired prefix

files_in_directory = os.listdir(directory)
morning_afternoons = [filename for filename in files_in_directory if filename.startswith(prefix)]

# PLOT 1
fig, ax = plt.subplots(1,2,figsize=(20, 7),gridspec_kw={'width_ratios': [4, 1]}, sharey=True, layout='constrained')
hourly_data = pd.read_csv('stats/run22_11579_results.csv', index_col='date')
hourly_data = hourly_data.drop(columns=['Unnamed: 0'])
hourly_data.index = pd.to_datetime(hourly_data.index)
hourly = ax[0].scatter(hourly_data.index, hourly_data['percentage agreement average'], label='average', alpha=0.3, color='red', marker='x')
ax[1].hist(hourly_data['percentage agreement average'], orientation='horizontal', color='red', alpha=0.3, label='Hourly',  weights=np.ones_like(hourly_data['percentage agreement average'])*100 / len(hourly_data['percentage agreement average']))
ax[0].set_ylabel('% Agreement')
ax[1].set_xlabel('Frequency')
ax[0].set_ylim([-10, 110]) 
ax[0].set_xlabel('Date')
ax[0].set_xticks([pd.to_datetime('2018'), pd.to_datetime('2019'), pd.to_datetime('2020-01-01'), pd.to_datetime('2021-01-01'), pd.to_datetime('2022-01-01'), pd.to_datetime('2023-01-01')])
ax[0].set_xticklabels(['2018', '2019', '2020', '2021', '2022', '2023'])
# plt.savefig('images/dcgan_pa_avg_hourly.pdf')


# PLOT 2
fig, ax = plt.subplots(1,2,figsize=(20, 7),gridspec_kw={'width_ratios': [4, 1]}, sharey=True, layout='constrained')
pa_sum_plot = {}
pa_avg_plot = {}
corr_sum_plot = {}
corr_avg_plot = {}
for filename in morning_afternoons:
 data = pd.read_csv(directory+filename, index_col='date')
 data = data.drop(columns=['Unnamed: 0'])
 data['category'] = [data.index[i].split('-')[-1] for i in range(len(data))]
 data['time'] = data['category'].apply(lambda x: 10 if x == 'morning' else 18)
 data['month'] = [data.index[i].split('-')[1] for i in range(len(data))]
 data['day'] = [data.index[i].split('-')[2] for i in range(len(data))]
 data['year'] = [data.index[i].split('-')[0] for i in range(len(data))]
 data['plot_date'] = pd.to_datetime(data['month'] + '-' + data['day'] + '-' + data['year']+ ' ' + data['time'].astype(str))
 data = data.drop(columns=['month', 'day', 'year', 'time'])
 morning = ax[0].scatter(data['plot_date'], data['percentage agreement average'], label='average', alpha=0.3, color='blue', marker='x')
 for i in range(len(data)):
  pa_sum_plot[data['plot_date'][i]] = data['percentage agreement sum'][i]
  pa_avg_plot[data['plot_date'][i]] = data['percentage agreement average'][i]
  corr_sum_plot[data['plot_date'][i]] = data['correlation sum'][i]
  corr_avg_plot[data['plot_date'][i]] = data['correlation average'][i]
ax[0].set_xlabel('Date')
# normalise the data before histogram
hist_data = pd.DataFrame.from_dict(pa_avg_plot, orient='index')
ax[1].hist(hist_data, orientation='horizontal', color='blue', alpha=0.4, label='Morning/Afternoon',  weights=np.ones_like(hist_data)*100 / len(hist_data))
ax[1].set_xlabel('Frequency')
ax[0].set_ylim([-10, 110])
ax[0].set_ylabel('% Agreement')
ax[0].set_xticks([pd.to_datetime('2018'), pd.to_datetime('2019'), pd.to_datetime('2020-01-01'), pd.to_datetime('2021-01-01'), pd.to_datetime('2022-01-01'), pd.to_datetime('2023-01-01')])
ax[0].set_xticklabels(['2018', '2019', '2020', '2021', '2022', '2023'])
# plt.savefig('images/dcgan_pa_avg_MA.pdf')



# now for the impact
MA_corr_avg = pd.DataFrame.from_dict(corr_avg_plot, orient='index')
MA_corr_sum = pd.DataFrame.from_dict(corr_sum_plot, orient='index') 
#first the corrolations
corr_avg = hourly_data['correlation average']
# count number of nans
nan_avg = np.count_nonzero(np.isnan(corr_avg))
corr_sum = hourly_data['correlation sum']
nan_sum = np.count_nonzero(np.isnan(corr_sum))

# PLOT 3
fig, ax = plt.subplots(1,2,figsize=(20, 10),gridspec_kw={'width_ratios': [3, 1]}, sharey=True, layout='constrained')
avg=ax[0].scatter(hourly_data.index, corr_avg, label='average', alpha=0.3, color='red', marker='x')
sum=ax[0].scatter(hourly_data.index, corr_sum, label='sum', alpha=0.2, color='blue', marker='x')
ax[1].hist(corr_avg, alpha=0.3, color='red', label='average', weights=np.ones_like(corr_avg)*100 / len(corr_avg), orientation='horizontal')
ax[1].hist(corr_sum, alpha=0.3, color='blue', label='sum', weights=np.ones_like(corr_sum)*100 / len(corr_sum), orientation='horizontal')
ax[1].set_xlabel('Frequency')
ax[0].set_xlabel('Date')
plt.ylabel('Correlation coefficient (R)')
fig.legend(handles=[avg, sum], labels=['Average', 'Sum'], loc='outside upper right', markerscale=2)
plt.savefig('images/dcgan_hourly_corr.pdf')

# PLOT 4
fig, ax = plt.subplots(1,2,figsize=(20, 10),gridspec_kw={'width_ratios': [3, 1]}, sharey=True, layout='constrained')
ma_sum=ax[0].scatter(MA_corr_sum.index, MA_corr_sum, label='MA sum', alpha=0.3, color='red', marker='o')
hourly=ax[0].scatter(hourly_data.index, corr_sum, label='sum', alpha=0.2, color='blue', marker='x')
ax[1].hist(corr_sum, alpha=0.3, color='blue', label='sum', weights=np.ones_like(corr_sum)*100 / len(corr_sum), orientation='horizontal')
ax[1].hist(MA_corr_sum, alpha=0.3, color='red', label='MA sum', weights=np.ones_like(MA_corr_sum)*100 / len(MA_corr_sum), orientation='horizontal')
ax[1].set_xlabel('Frequency')
ax[0].set_xlabel('Date')
plt.ylabel('Correlation coefficient (R)')
fig.legend(handles=[ma_sum, hourly], labels=['Morning/Afternoon', 'Hourly'], loc='outside upper right', markerscale=2)
plt.savefig('images/sum_corr_hourlyvsmorn.pdf')


# PLOT 5
fig, ax = plt.subplots(1,2,figsize=(20, 10),gridspec_kw={'width_ratios': [3, 1]}, sharey=True, layout='constrained')
hourly=ax[0].scatter(hourly_data.index, corr_avg, label='sum', alpha=0.2, color='blue', marker='x')
ma_avg=ax[0].scatter(MA_corr_avg.index, MA_corr_avg, label='MA average', alpha=0.3, color='red', marker='o')
ax[1].hist(corr_avg, alpha=0.3, color='blue', label='average', weights=np.ones_like(corr_avg)*100 / len(corr_avg), orientation='horizontal')
ax[1].hist(MA_corr_avg, alpha=0.3, color='red', label='MA average', weights=np.ones_like(MA_corr_avg)*100 / len(MA_corr_avg), orientation='horizontal')
ax[1].set_xlabel('Frequency')
ax[0].set_xlabel('Date')
plt.ylabel('Correlation coefficient (R)')
fig.legend(handles=[ma_avg, hourly], labels=['Morning/Afternoon', 'Hourly'], loc='outside upper right', markerscale=2)
plt.savefig('images/avg_corr_hourlyvsmorn.pdf')

# PLOT 6
arr_corr_avg = corr_avg.to_numpy()
arr_corr_sum = corr_sum.to_numpy()
arr_MA_corr_avg = MA_corr_avg[0].to_numpy()
arr_MA_corr_sum = MA_corr_sum[0].to_numpy()
all_corr = [arr_corr_avg, arr_corr_sum, arr_MA_corr_avg, arr_MA_corr_sum]
fig, ax = plt.subplots(figsize=(15, 10))
ax.hist(all_corr,bins=15, histtype='bar', alpha=0.7, label=['Hourly average', 'Hourly sum', 'Morning/Afternoon average', 'Morning/Afternoon sum'], density=False)
plt.xlabel('Correlation Coefficient (R)')
plt.ylabel('Count')
plt.ylim([0, 1000])
plt.legend()
plt.tight_layout()
plt.savefig('images/dcgan_corr_histogram.pdf')


# PLOT 7
fig, ax = plt.subplots(figsize=(15, 10))
labels = ['Hourly average', 'Hourly sum', 'Morning/Afternoon average', 'Morning/Afternoon sum']
for i, result in enumerate(all_corr):
 ax.hist(result,bins=15, histtype='step', alpha=0.7,weights=np.ones_like(result)*100/len(result), label=labels[i] , fill=False, linewidth=5 )
plt.xlabel('Correlation Coefficient (R)')
plt.ylabel('Frequency (%)')
plt.legend()
plt.savefig('images/dcgan_corr_histogram_alternative.pdf')

# HYSPLIT ANALYSIS for comparison
directory = '../../HYSPLIT_results/CR_analysis/results/'  # Replace with the actual directory path
suffix = 'morningafternoon_results.csv'  # Replace with the desired prefix
files_in_directory = os.listdir(directory)
morning_afternoons = [filename for filename in files_in_directory if filename.endswith(suffix)]
hourly_files = os.listdir(directory+'hourly/')
hysplit_pa_sum_plot = {}
hysplit_pa_avg_plot = {}
hysplit_corr_sum_plot = {}
hysplit_corr_avg_plot = {}

hysplit_hr_pa_sum_plot = {}
hysplit_hr_pa_avg_plot = {}
hysplit_hr_corr_sum_plot = {}
hysplit_hr_corr_avg_plot = {}
for filename in hourly_files:
 data = pd.read_csv(directory+'hourly/'+filename, index_col='date')
 data.index = pd.to_datetime(data.index, format='%Y-%m-%d-%H')
 for i in range(len(data)):
  hysplit_hr_pa_sum_plot[data.index[i]] = data['percentage_agreement_sum'][i]
  hysplit_hr_pa_avg_plot[data.index[i]] = data['percentage_agreement_avg'][i]
  hysplit_hr_corr_sum_plot[data.index[i]] = data['corr_sum'][i]
  hysplit_hr_corr_avg_plot[data.index[i]] = data['corr_avg'][i]


for filename in morning_afternoons:
 data = pd.read_csv(directory+filename, index_col='date')
 data['category'] = [data.index[i].split('-')[-1] for i in range(len(data))]
 data['time'] = data['category'].apply(lambda x: 10 if x == 'morning' else 18)
 data['month'] = [data.index[i].split('-')[1] for i in range(len(data))]
 data['day'] = [data.index[i].split('-')[2] for i in range(len(data))]
 data['year'] = [data.index[i].split('-')[0] for i in range(len(data))]
 data['plot_date'] = pd.to_datetime(data['month'] + '-' + data['day'] + '-' + data['year']+ ' ' + data['time'].astype(str))
 data = data.drop(columns=['month', 'day', 'year', 'time'])
 for i in range(len(data)):
  hysplit_pa_sum_plot[data['plot_date'][i]] = data['percentage_agreement_sum'][i]
  hysplit_pa_avg_plot[data['plot_date'][i]] = data['percentage_agreement_avg'][i]
  hysplit_corr_sum_plot[data['plot_date'][i]] = data['corr_sum'][i]
  hysplit_corr_avg_plot[data['plot_date'][i]] = data['corr_avg'][i]

hysplit_MA_corr_avg = pd.DataFrame.from_dict(hysplit_corr_avg_plot, orient='index')
hysplit_MA_corr_sum = pd.DataFrame.from_dict(hysplit_corr_sum_plot, orient='index') 
#first the corrolations
hysplit_corr_avg = pd.DataFrame.from_dict(hysplit_hr_corr_avg_plot, orient='index')
hysplit_corr_sum = pd.DataFrame.from_dict(hysplit_hr_corr_sum_plot, orient='index')

hysplit_hr_pa_sum = pd.DataFrame.from_dict(hysplit_hr_pa_sum_plot, orient='index')
hysplit_hr_pa_avg = pd.DataFrame.from_dict(hysplit_hr_pa_avg_plot, orient='index')
hysplit_pa_sum = pd.DataFrame.from_dict(hysplit_pa_sum_plot, orient='index')
hysplit_pa_avg = pd.DataFrame.from_dict(hysplit_pa_avg_plot, orient='index')

# get the dcgan one
pa_sum = pd.DataFrame.from_dict(pa_sum_plot, orient='index')
pa_avg = pd.DataFrame.from_dict(pa_avg_plot, orient='index')


def kl_divergence(p, q):
    vals = []
    for i in range(len(p)):
        if q[i] == 0 or p[i] == 0:
            # Handle special cases to avoid division by zero or negative log
            print('zeros', p[i], q[i])
            continue
        val = p[i] * np.log2(p[i] / q[i])
        vals.append(val)
    return np.sum(vals)
 
def js_divergence(p, q):
 m = 0.5 * (p + q)
 print('m', m)
 return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

def make_probs(dcgan, bins=20):
 """ Input is results from a models during an analsyis method. Eg. the results from the path
 analysis for average aggregation of smell reports. These arrays come from the dictionaries collected.
 To get in the correct format, you convert the dictionary to a df, then use df[0].to_numpy() to get the array.
 """
 # split into 20 bins
 # remove nans if there are
 dcgan = dcgan[~np.isnan(dcgan)]
 dcgan_vals, dcgan_bins = np.histogram(dcgan, bins=bins, density=True)
 return dcgan_vals

# ok lets do this for all the comparisons
dcgan_list = [pa_sum[0].to_numpy(), pa_avg[0].to_numpy(), 
              corr_sum.to_numpy(), corr_avg.to_numpy(),
              MA_corr_sum[0].to_numpy(), MA_corr_avg[0].to_numpy()]
hysplit_list = [hysplit_pa_sum[0].to_numpy(), hysplit_pa_avg[0].to_numpy(),
                hysplit_corr_sum[0].to_numpy(), hysplit_corr_avg[0].to_numpy(),
                hysplit_MA_corr_sum[0].to_numpy(), hysplit_MA_corr_avg[0].to_numpy()]
names = ['PA sum', 'PA avg', 'Corr sum', 'Corr avg', 'MA Corr sum', 'MA Corr avg']
for dcgan, hysplit, name in zip(dcgan_list, hysplit_list, names):
 js_dist = distance.jensenshannon(make_probs(dcgan, bins=40), make_probs(hysplit, bins=40))
 js_divergence = js_dist**2
 print(f'JS divergence: {js_divergence.round(5)} for {name}' )


# now comparing morning and hourly for each model's correlation analysis
hourly = [corr_sum.to_numpy(), corr_avg.to_numpy(), hysplit_corr_sum[0].to_numpy(), hysplit_corr_avg[0].to_numpy()]
MA = [MA_corr_sum[0].to_numpy(), MA_corr_avg[0].to_numpy(), hysplit_MA_corr_sum[0].to_numpy(), hysplit_MA_corr_avg[0].to_numpy()]
names = ['dcgan Corr sum', 'dcgan Corr avg', 'hysplit Corr sum', 'hysplit Corr avg']

for hourly, MA, name in zip(hourly, MA, names):
 js_dist = distance.jensenshannon(make_probs(hourly, bins=40), make_probs(MA, bins=40))
 js_divergence = js_dist**2
 print(f'JS divergence: {js_divergence.round(5)} for {name}' )

