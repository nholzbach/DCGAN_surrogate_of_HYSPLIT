# This code is used to preprocess and explore the sensor data. Then, plots the emission scheme for each station. Finally, missing data is 
# imputed and this can be used for the informed input vector for the DCGAN.
# A number of plots are generated for each monitoring station sensor:
# 1. Monthly patterns
# 2. Seasonal patterns
# 3. Average emission per hour for each station, ie. the emission scheme for HYSPLIT
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from geopy import distance
from os.path import isfile, join
from os import listdir
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("talk")
from datetime import datetime

# this may need to be edited depending on where the sensor data is stored
PATH = "sensor_data/"

def preprocess_sensor(df):
  df.index = pd.to_datetime(df.index, unit="s", utc=True)
  df = df.rename(columns={df.columns[0]: 'emission'})
  df['Year'] = df.index.year
  df['Month'] = df.index.month
  df['Day'] = df.index.day
  df['Hour'] = df.index.hour
  df = df.resample("60Min", label="right").mean()
  df['Season'] = df['Month'].apply(lambda x: 'Winter' if x in [12,1,2] else ('Spring' if x in [3,4,5] else ('Summer' if x in [6,7,8] else 'Autumn')))
  
  result = df.groupby([df['Year'], df['Month'], df['Hour'] ])['emission'].mean().round(5)
  season_result = df.groupby([df['Year'], df['Season'], df['Hour'] ])['emission'].mean().round(5)
  season_result_std = df.groupby([df['Year'], df['Season'], df['Hour'] ])['emission'].std().round(5)
  hour_result = df.groupby('Hour')['emission'].mean().round(5)
  return df, result, season_result, season_result_std, hour_result

# Plots 1 and 2
def pattern_plots(df, name, result, season_result, season_result_std, hour_result, save =True):
  # monthly patterns
  station = name
  unique_years = df['Year'].dropna().unique()
  years = unique_years
  
  fig1, axs = plt.subplots(len(years),1, figsize=(10,20), sharex=True, sharey=True)
  axs[3].set_ylabel('PM2.5 ($\mu$g/m$^3$)') 
  
  for i, year in enumerate(years):
    ax = axs[i]
    conc_year = result[year]
    # remove .0 from year
    ax.set_title(f'Year {int(year)}')
    months = conc_year.index.get_level_values(0).unique().tolist()
    # for month in range(1,13):
    for month in months:
      conc_year[month].plot(ax=ax, label=month)

  axs[-1].legend()

  
  # Seasonal patterns
  fig2, axs = plt.subplots(len(years), 1, figsize=(10, 20), sharex=True, sharey=True)
  for i, year in enumerate(years):
      ax = axs[i]
      conc_year = season_result[year]
      conc_std = season_result_std[year]
      ax.set_title(f'Year {int(year)}')
      seasons = conc_year.index.get_level_values(0).unique().tolist() 
      for season in seasons:
          y_mean = conc_year[season]
          y_std = conc_std[season]
          
          ax.plot(y_mean.index, y_mean.values, label=season)
          ax.fill_between(y_mean.index, y_mean.values - y_std.values, y_mean.values + y_std.values, alpha=0.1)
      
  axs[-1].legend()

  plt.xlabel('Hour')
  plt.ylabel('PM2.5 ($\mu$g/m$^3$)')
  plt.tight_layout()
  plt.show()
  fig1.savefig(f"../figures/monthly_{station}.png")
  fig2.savefig(f"../figures/seasonal_{station}.png")
  
  # Hourly patterns
  plt.plot(hour_result.index, hour_result, marker='o', linestyle='-')

  # Set the x-axis and y-axis labels
  plt.xlabel('Hour')
  plt.ylabel('Average PM 2.5 Emission ($\mu$g/m$^3$)')

  # Set the title of the plot
  plt.title('Average Emission per hour over all years')

  # Display the plot
  plt.show()
  
def export_to_csv(hour_result, name):
  hour_result.index = hour_result.index +1
  pd.DataFrame(hour_result).to_csv(PATH+'%s.csv' % name)

def monitor_process(data, name, save = False):
 df, result, season_result, season_result_std, hour_result = preprocess_sensor(data) 
 pattern_plots(df, name, result, season_result, season_result_std, hour_result, save =False)
 if save == True:
   export_to_csv(hour_result, name)


# EDIT THIS if you want to run for different stations
PATH = "../sensor_data/"
names = ['NorthBraddock_emission','Glassport_emission','Liberty_emission', 'Harrison_emission', 'Lawrenceville_emission' ]
file_names = ['Feed_3_North_Braddock_ACHD_PM10.csv', 
              'Feed_24_Glassport_High_Street_ACHD.csv', 
              'Feed_29_Liberty_2_ACHD_PM25.csv',
              'Feed_25_Harrison_Township_ACHD.csv',
              'Feed_26_and_Feed_59665_Lawrenceville_ACHD_PM25.csv' ]

for file, name in zip(file_names, names):
  print(f"---------------{file}---------------")
  df = pd.read_csv(PATH+file).set_index("EpochTime")
  monitor_process(df, name, save=True)

 
# PLOT 3
def all_emission_used_plot(names):
  """plot average emission per hour for each station"""
  
  fig, axs = plt.subplots(figsize=(14,10))
 
  for i, name in enumerate(names):
    # print(PATH+name+'.csv')
    df = pd.read_csv(PATH+name+'.csv')
    # print(df.head())
    x = df['Hour']
    y = df['emission']
    # hour_result = station[hour_result]
    if name == 'Harrison_emission':
      axs.plot(x,y*1000, marker='o',linestyle='-', label = name.split('_')[0]+'x1000')
    else:
      axs.plot(x,y, marker='o', linestyle='-', label = name.split('_')[0])
  plt.legend(fontsize = 15)
  # ticks are 0 - 24 
  plt.xticks(np.arange(0,25,2),fontsize=15)
  plt.yticks(fontsize=15)
  plt.xlabel('Hour', fontsize=20)
  plt.ylabel('PM2.5 Emission ($\mu g/m^3$)', fontsize=20)
  plt.savefig('../figures/emission_schemes.png')

all_emission_used_plot(names)


# filling missing data
def preprocess_sensor_missing(file, name, save=False):
  df = pd.read_csv(PATH+file).set_index("EpochTime")
  df.index = pd.to_datetime(df.index, unit='s', utc=True)
  df = df.resample("60Min", label="right").mean()
  df = df.rename(columns={df.columns[0]: 'emission'})
  
  # check for nans
  new_dates = pd.date_range(start='2016-10-31 06:00:00+00:00', end='2022-12-13 00:00:00+00:00', freq='H')
  extended = df.reindex(new_dates)
  # count number of negatives:
  neg_count = np.sum((extended < 0).values.ravel())
  print('number of negative values:', neg_count)
  print("pre-imputation min and max emissions:", extended.max(), extended.min())
  filled = extended.interpolate(method='time', limit=2)  
  
  nan_count = filled['emission'].isna().sum()
  print("Iniital count of NaN entries in 'emission':", nan_count)
  
  nan_dates = filled['emission'].isna().groupby([filled.index.year, filled.index.month, filled.index.day]).sum()
  missing_dates = nan_dates[nan_dates > 0].index
  full_days = []
  partial_days = []

  for date in missing_dates:
    total_missing_hours = nan_dates[date]

    if total_missing_hours > 19:
      full_days.append(date)
      day = datetime(*date).strftime('%Y-%m-%d')
      # full with -1
      filled.loc[filled.index.astype(str).str.contains(day),'emission'] = -1
    else:
      partial_days.append(date)
      day = datetime(*date).strftime('%Y-%m-%d')
      # fill with mean top bottom
      filled.loc[filled.index.astype(str).str.contains(day),'emission'] = (filled.loc[filled.index.astype(str).str.contains(day),'emission'].shift() + filled.loc[filled.index.astype(str).str.contains(day),'emission'].shift(-1))/2
  
  print(f"full days:{len(full_days)}, partial days: {len(partial_days)}")
  
  # the rest can actually just be filled with -1 too
  print("last few nans that can't be inputed", filled['emission'].isna().sum())  
  filled['emission'] = filled['emission'].fillna(-1)
  print("check that there's no more missing:", filled['emission'].isna().sum()) 

  
  if save == True:
    # save as csv
    filled.to_csv(PATH+name+'_filled-1.csv')


for name, file in zip(names, file_names):
  print(f"---------------{name}---------------")
  preprocess_sensor_missing(file, name, save=True)
  
  




