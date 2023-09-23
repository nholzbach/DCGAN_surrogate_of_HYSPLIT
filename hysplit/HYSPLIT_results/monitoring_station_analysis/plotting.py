# This file does the dixed point analysis with each of the monitoring stations.
# The plots generated are:
# 1. RMSE for each station
# 2. Correlation coefficient (R) for each station
# 3. Histogram of R values for each station (distribution of R values)
# 4. Comparison of predicted and observed values for each station for full time period
# These plots save to plots/ folder
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, sys
import seaborn as sns
from datetime import datetime
sns.set(style="whitegrid")
sns.set_context("talk", font_scale=2.5,rc={"lines.linewidth": 2})
# load data from files. Want 5 dataframes (one for each station) with columns: time, rmse, r, p, predictions, observations
libertyframe = {}       
glassportframe = {}
harrisonframe = {}
northbraddockframe = {}
lawrencevilleframe = {}

# CHANGE THIS PATH TO WHEREVER THE RESULTS ARE STORED
folder_path = 'results/negative_test/'
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):  # Specify the file extension if needed

        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path, index_col='Unnamed: 0')
        date = filename.split('.')[0]
        yr, mo, day = date.split('_')[0:3]
        date_entry = pd.to_datetime(f'{yr}-{mo}-{day}')
        libertyframe[date_entry] = df['Liberty']
        glassportframe[date_entry] = df['Glassport']
        harrisonframe[date_entry] = df['Harrison']
        northbraddockframe[date_entry] = df['NorthBraddock']
        lawrencevilleframe[date_entry] = df['Lawrenceville']

libertydf = pd.DataFrame(libertyframe)
glassportdf = pd.DataFrame(glassportframe)
harrisondf = pd.DataFrame(harrisonframe)
northbraddockdf = pd.DataFrame(northbraddockframe)
lawrencevilledf = pd.DataFrame(lawrencevilleframe)

# Plots 1,2 
def stats(measure, station, name,i, plot = True):
    grouped_df = station.groupby([station.columns.year, station.columns.month], axis=1)
    monthly_stats = {}
    for date, month_group in grouped_df:
        vals = month_group.loc[measure].astype(float)
        stats = [vals.mean(), vals.std()]
        monthly_stats[date] = stats
        if plot:
            ax[i].set_title(f'{name}')
            vals.plot(ax = ax[i], color='grey')
            # plot with dots
            vals.plot(ax = ax[i], marker='o', linestyle='none', color='black', markersize=5, alpha=0.5)
     
    return monthly_stats

for measure in ['rmse', 'r']: 
    fig, ax = plt.subplots(figsize=(20,10), sharex=True)      

    plt.xlabel('Date')
    for station, name in zip([libertydf, glassportdf, harrisondf, northbraddockdf, lawrencevilledf],['Liberty', 'Glassport', 'Harrison', 'North Braddock', 'Lawrenceville']):
        monthly_stats_list = [stats(measure, station,name,0, plot = False)]
        for monthly_stats in monthly_stats_list:
            x = [datetime(year, month,1) for year, month in monthly_stats.keys()]
            means = [stat[0] for stat in monthly_stats.values()]
            std = [stat[1] for stat in monthly_stats.values()]

            ax.scatter(x, means, label = name)
            ax.plot(x, means)
            ax.fill_between(x, np.array(means)-np.array(std), np.array(means)+np.array(std), alpha=0.1) 
    if measure == 'rmse':
        ax.set_ylim([0,70])
        plt.legend()
    else:

        ax.set_ylim([-1,1])
        
    plt.savefig(f'plots/{measure}_meansonly_all_stations_hysplit.pdf')


# Plot 3
all_r = []
fig,ax = plt.subplots(figsize=(15,10))
for station, name in zip([libertydf, glassportdf,northbraddockdf, lawrencevilledf],['Liberty', 'Glassport', 'North Braddock', 'Lawrenceville']):
    grouped_df = station.groupby([station.columns.year, station.columns.month], axis=1)
    rs = []
    for date, month_group in grouped_df:
        vals = month_group.loc['r'].astype(float)
        rs.append(vals.values)
    rs = np.concatenate(rs)
    all_r.append(rs)
    # print(rs)
    sns.histplot(rs, kde=True, binwidth=0.05, binrange=(-1,1), alpha=0.3, label = name)
plt.xlabel('Correlation Coefficient')
plt.legend()
plt.tight_layout()
plt.savefig('plots/hysplit_r_hist.pdf')
np.savetxt('plots/r_values.txt', all_r)



# PLOT 4
def convert_str_to_array(df):
    if type(df) == str:
        data_str = df.replace('[', '').replace(']', '')
    numpy_array = np.fromstring(data_str, sep=' ')
    return numpy_array

def extract_flat(df, i):
    df = df[i]
    if type(df) == float:
        arr = 0.0
        return arr
    if type(df) == str:
        arr = convert_str_to_array(df)
        return arr.mean()
    

def to_plot(col, date, month_group):
    df = month_group.loc[col]
    df = df.sort_index()
    df_list = []
    if date[1] in [1, 3, 5, 7, 8, 10, 12]:
        times = pd.date_range(start=f'{date[0]}-{str(date[1]).zfill(2)}-01', end=f'{date[0]}-{str(date[1]).zfill(2)}-31 23:00', freq='D')
        df = df.reindex(times)
        df = df.fillna(0.0)
        
        for i in range(31):
            array = extract_flat(df, i)
            if array == None:
                array = 0.0
            
            df_list.append(array)
        flat = np.array(df_list)
    elif date[1] in [4,6,9,11]:
        times = pd.date_range(start=f'{date[0]}-{str(date[1]).zfill(2)}-01', end=f'{date[0]}-{str(date[1]).zfill(2)}-30 23:00', freq='D')
        df = df.reindex(times)
        df = df.fillna(0.0)
        for i in range(30):
            array = extract_flat(df, i)

            df_list.append(array)
        flat = np.array(df_list)
    else:
        times = pd.date_range(start=f'{date[0]}-{str(date[1]).zfill(2)}-01', end=f'{date[0]}-{str(date[1]).zfill(2)}-28 23:00', freq='D')
        df = df.reindex(times)
        df = df.fillna(0.0)
        for i in range(28):
            array = extract_flat(df, i)

            df_list.append(array)
        flat = np.array(df_list)

    return flat, times


sns.set(style="whitegrid")
sns.set_context("talk", font_scale=2.5,rc={"lines.linewidth": 3})
def plot_station_comparison(station, name, i, save =False):
    grouped_df = station.groupby([station.columns.year, station.columns.month], axis=1)
    for date, month_group in grouped_df:
        if date[0] == 2016 and (date[1] == 11 or date[1] == 12):
            print("passing", date)
            pass
        else:
            for col in ['predictions','targets']:
                flat, times = to_plot(col, date, month_group)
                if col == 'predictions':
                    if name == 'Harrison':
                        ax[i].plot(times, flat, 'b', alpha=0.6)
                        ax[i].set_ylim([-0.001, 0.02])
                    else:
                        ax[i].plot(times, flat*1000, 'b', alpha=0.6)
                else:
                    ax[i].plot(times, flat, 'r', alpha=0.6)
    ax[i].set_title(f'{name}')

fig, ax = plt.subplots(5,1, figsize=(40,20), sharex=True, layout='constrained' ) 

plot_station_comparison(libertydf, 'Liberty',0)
plot_station_comparison(glassportdf, 'Glassport',2)
plot_station_comparison(harrisondf, 'Harrison',3)
plot_station_comparison(northbraddockdf, 'North Braddock',4)
plot_station_comparison(lawrencevilledf, 'Lawrenceville',1)
plt.legend(['Predicted value x1000', 'Observed value'])
ax[2].set_ylabel('PM2.5 ($ \mu g/m^3$)')
leg = plt.legend(['Predicted value x1000', 'Observed value'])
for legobj in leg.legend_handles:
    legobj.set_linewidth(5.0)
plt.savefig(f'plots/all_stations_comparison.pdf')

