# This code is for the analysis of the citizen reports and the DCGAN output. There are two parts to it, and you might not want to run it all. 
# From line 171 onwards, read the comments and comment out what you do not want to do. 
# First you can specify a date and look at the results for that date, this can be done next for a morning/afternoon resolution for a particular month
# 
# Second, the analysis is done at a morning/afternoon resolution for a sightly extended time range. In this case, it's a collection of days that are known to visually align well.
# Adjust the days of interest in the code. 
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


def plot_grids(grid, cr_grid, date, measure):
 fig , ax = plt.subplots(1,2, figsize = (15,15), sharex=True, sharey=True)
 # plot the two grids
 ax[0] = grid.plot(column='emission', ax = ax[0], cmap = 'rocket_r')
 ax[0].set_title('DCGAN emissions results')
 sm = plt.cm.ScalarMappable(cmap = 'rocket_r')
 sm.set_array(grid['emission'])
 cbar = plt.colorbar(sm, ax=ax[0], shrink=0.4)

 ax[1]= cr_grid.plot(column='smell value', ax=ax[1], cmap = 'rocket_r')
 ax[1].set_title('Citizen reports')
 sm2 = plt.cm.ScalarMappable(cmap = 'rocket_r')
 sm2.set_array(cr_grid['smell value'])
 cbar = plt.colorbar(sm2, ax=ax[1], shrink=0.4) 
 plt.show()
 fig.savefig(f'images/grid_comparison_{date}_{measure}.pdf')


def boundary_setup(n_cells):
    xmin, ymin, xmax, ymax = -80.1892, 40.2246, -79.6912, 40.5963
    cell_size_x = (xmax - xmin) / n_cells
    cell_size_y = (ymax - ymin) / n_cells
    crs = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs"
    grid_cells = []
    for x0 in np.arange(xmin, xmax, cell_size_x):
        for y0 in np.arange(ymin, ymax, cell_size_y):
            x1 = x0 + cell_size_x
            y1 = y0 + cell_size_y
            grid_cells.append(shapely.geometry.box(x0, y0, x1, y1))
    cell = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=crs)

    bbox = shapely.geometry.box(xmin, ymin, xmax, ymax)
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs=crs)
    return cell, bbox_gdf, crs

def preprocess(df, bbox_gdf, crs):
 geometry = [Point(xy) for xy in zip(df['skewed longitude'], df['skewed latitude'])]
 gdf = gpd.GeoDataFrame(df, geometry=geometry)
 gdf = gdf.drop(['skewed longitude', 'skewed latitude'], axis=1)
 gdf.crs = crs

 points_in_bbox = gpd.sjoin(gdf, bbox_gdf, how="inner",predicate='within')
 joined_df = points_in_bbox.drop(columns=['index_right'])
 return joined_df


# these two functions can definitely be made into one - ran out of tim
def merge_sum(result_gdf, cell,date,group=None, plot = False):
    merged = gpd.sjoin(result_gdf, cell, how='left', predicate='within')
    cell_max_smell = merged.groupby('index_right')['smell value'].sum()
    cell = cell.merge(cell_max_smell, how='left', left_index=True, right_index=True)
    cell['smell value'] = cell['smell value'].fillna(0)
    if plot:
        ax = cell.plot(column='smell value', figsize=(12, 8), cmap='BuGn', edgecolor="white")
        sm = plt.cm.ScalarMappable(cmap='BuGn')
        sm.set_array(cell['smell value'])
        
        norm = mcolors.Normalize()
        sm.set_norm(norm)
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Smell Value')
        ax.set_title(f'Summed smell value for {date}')
    return cell

def merge_average(result_gdf, cell,date, group=None, plot = False):
    merged = gpd.sjoin(result_gdf, cell, how='left', predicate='within')
    cell_max_smell = merged.groupby('index_right')['smell value'].mean()
    cell = cell.merge(cell_max_smell, how='left', left_index=True, right_index=True)
    cell['smell value'] = cell['smell value'].fillna(0)
    if plot:
        ax = cell.plot(column='smell value', figsize=(12, 8), cmap='BuGn', edgecolor="white")

        sm = plt.cm.ScalarMappable(cmap='BuGn')
        sm.set_array(cell['smell value'])
        
        norm = mcolors.Normalize(vmin=0, vmax=5)
        sm.set_norm(norm)
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Smell Value') 
        ax.set_title(f'Average smell value for {date}')
    return cell

def make_dcgan_gdf(array, geoms):
    try:
        grid = pd.DataFrame(array)
        grid.columns = ['emission']
        grid['geometry'] = geoms
        geo_df = gpd.GeoDataFrame(grid, geometry='geometry')
        geo_df.crs = 'EPSG:4326'
    except (ValueError, AttributeError):
        print('geo_df not working for some reason so skipping this one')
        geo_df = None
    return geo_df
    

def get_impact_stats(df, date, plot=False, title=None, plotsave=False):
 if df is None:
    print('no matching data for this time')
    return None, None
 else:
    df.drop(columns=['geometry'], inplace=True)

    means = df.groupby('smell value').mean()
    means.rename(columns={'emission': 'Mean'}, inplace=True)
    
    std = df.groupby('smell value').std()
    std.rename(columns={'emission': 'STD'}, inplace=True)
    
    counts = df.groupby('smell value').count()
    counts.rename(columns={'emission': 'Count'}, inplace=True)
    
    corr = df.corr()
    print(corr)
    
    stats = pd.concat([means, counts, std], axis=1)
    
    if plot == True:
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.plot(df['smell value'], df['emission'], 'rx', alpha=0.5, markersize=20, markeredgewidth=6)
        plt.plot(stats.index, stats['Mean'], linewidth=6)
        plt.fill_between(stats.index, stats['Mean']-stats['STD'], stats['Mean']+stats['STD'], alpha=0.2)
        if title == 'Average':
            plt.xlabel('Average smell value')
        elif title == 'Sum':
            plt.xlabel('Summed smell value')
        plt.ylabel('PM2.5 concentration $\mu g/m^3$')
        plt.tight_layout()
    if plotsave == True:
        plt.savefig(f'images/impact_analysis_{date}_{title}.pdf')
    

    return stats, corr['emission']['smell value']

def percentage_agreement(hysplit, cr):
    cr['smell value'] = cr['smell value'].replace(0, np.nan)
    df1 = cr.dropna()
    try:
        merged_df = df1.merge(hysplit, on='geometry', suffixes=('_df1', '_df2'))
    except ValueError:
        print('value error')
        return 0, None

    nonzero_vals = len(merged_df.where(merged_df['emission'] > 0).dropna())
    print('nonzero vals', nonzero_vals)

    total_values = len(df1)
    print('total values', total_values)
    pa = (nonzero_vals / total_values) * 100
    return pa, merged_df 


# ADJUST VARIABLES 
test_num = 22
dcganPATH = f'analysis/grids/run{test_num}/'
input_info = torch.load(f'../results_images/test_{test_num}/state11579_full_input.pt')

dates = []
for i in range(len(input_info)):
    dates.append(input_info[i][-9])
# inverse scale
dates = np.array(dates)
dates = dates * (1680300000 - 1470002400) + 1470002400
dates = pd.to_datetime(dates, unit='s')
dcgan_output = []
with h5py.File(f'../results_images/test_{test_num}/tensors_state11579.h5', 'r') as hf:
    for dataset_name in hf:
        dcgan_output.append(hf[dataset_name][:])

full={}
# scale the dcgan output
for i in range(len(dcgan_output)):
    grid = dcgan_output[i].squeeze()
    flat_grid = grid.flatten('F')
    flat_grid_rescaled = (flat_grid - (-1.0)) * (35.0 - 0.0) / (1.0 - (-1.0)) + 0.0
    full[dates[i]] = flat_grid_rescaled
    
sorted_tuples = sorted(zip(dates, dcgan_output))
sorted_date_dcgan = [(date, dcgan) for date, dcgan in sorted_tuples]
sorted_dcgan = [dcgan for date, dcgan in sorted_tuples]
sorted_dates = [date for date, dcgan in sorted_tuples]

df = pd.DataFrame.from_dict(full, orient='index')
df.index = pd.to_datetime(df.index)
df = df.sort_index()

# load all the cr grids
smell_reports = pd.read_csv('../../../citizen_data/raw/smell_reports.csv', usecols=["epoch time","skewed latitude", "skewed longitude","smell value"]).set_index("epoch time")
smell_reports.index = pd.to_datetime(smell_reports.index, unit='s')
# grouped at daily resolution
cell, bbox, crs = boundary_setup(64)
grouped_by_hour = smell_reports.groupby(smell_reports.index.floor('H'))

# MAY NEED TO CHANGE THIS ACCORDING TO FILE STRUCTURE
geoms = pickle.load(open('../../HYSPLIT_results/geoms.pkl', 'rb'))

# SPECIFY THE DATETIME YOU WANT TO LOOK AT and PLOT (hourly resolution)
interested_date = '2022-10-22-22'
results = {}
for dayhour, rows in tqdm(grouped_by_hour):
    datehr = dayhour.strftime('%Y-%m-%d-%H')
    if datehr == interested_date:
        print('match', dayhour)
        index = sorted_dates.index(dayhour)
        print('index:', index)
        print('date:', sorted_dates[index])
        group_df = smell_reports.loc[rows.index]
        print('group_df', group_df)
        if len(group_df) == 0:
            print('no data for this hour')
            continue
        
        else:
            result_gdf = preprocess(group_df, bbox, crs)
            merged_avg = merge_average(result_gdf, cell, dayhour, plot=False)
            merged_sum = merge_sum(result_gdf, cell, dayhour, plot=False)

            dcgan = make_dcgan_gdf(df.iloc[index], geoms)


        # plot all
        # fig, ax = plt.subplots(1,3, figsize=(30,10), sharex=True, sharey=True)
        # merged_avg.plot(column='smell value', ax=ax[0], cmap='BuGn', edgecolor="white")
        # sm = plt.cm.ScalarMappable(cmap='BuGn')
        # sm.set_array(merged_avg['smell value'])
        # norm = mcolors.Normalize(vmin=0, vmax=5)
        # sm.set_norm(norm)
        # cbar = plt.colorbar(sm, ax=ax[0], shrink=0.5)
        # # cbar.set_label('Smell Value')
        # ax[0].set_title(f'Average smell value for {dayhour}')
        # merged_sum.plot(column='smell value', ax=ax[1], cmap='BuGn', edgecolor="white")
        # sm2 = plt.cm.ScalarMappable(cmap='BuGn')
        # sm2.set_array(merged_sum['smell value'])
        # cbar2 = plt.colorbar(sm2, ax=ax[1], shrink=0.5)
        # cbar2.set_label('Smell Value')
        # fig.tight_layout()
        # ax[1].set_title(f'Summed smell value for {dayhour}')
        # dcgan.plot(column='emission', cmap='BuGn', edgecolor="white", ax=ax[2])
        # ax[2].set_title(f'DCGAN output for {dayhour}')
        # sm3 = plt.cm.ScalarMappable(cmap='BuGn')
        # sm3.set_array(dcgan['emission'])
        # cbar3 = plt.colorbar(sm3, ax=ax[2], shrink=0.5)
        # cbar3.set_label('Emissions')
        
        # actual analysis
            pa_average, df_average = percentage_agreement(dcgan, merged_avg)
            pa_sum, df_sum = percentage_agreement(dcgan, merged_sum)
            print('average', pa_average)
            print('sum', pa_sum)
            impactstats_avg, corr_avg = get_impact_stats(df_average, datehr, plot=True, title='Average', plotsave=True)
            impactstats_sum, corr_sum = get_impact_stats(df_sum, datehr, plot=True, title='Sum', plotsave=True)


# Morning afternoon analysis
smell_reports['Year'] = smell_reports.index.year
smell_reports['Month'] = smell_reports.index.month
smell_reports['Day'] = smell_reports.index.day
smell_reports['Hour'] = smell_reports.index.hour
ranges = [range(6, 14 + 1), range(15, 22 + 1)]
names = ['morning', 'afternoon']
# DEFINE WHICH MONTH TO LOOK AT
yr, mo = 2022,10
month_results = {}
combined_df_average = pd.DataFrame()
smell_reports_range = smell_reports[(smell_reports['Year']==yr) & (smell_reports['Month']==mo)]
for day in range(22,23):
    for name, hours in zip(names, ranges):
        time = f'{yr}-{mo}-{day}-{name}'
        start = pd.Timestamp(f'{yr}-{str(mo).zfill(2)}-{str(day).zfill(2)}-{str(hours[0]).zfill(2)}')
        end = pd.Timestamp(f'{yr}-{str(mo).zfill(2)}-{str(day).zfill(2)}-{str(hours[-1]).zfill(2)}')
        mask = (df.index >= start) & (df.index <= end)
        subset = df.loc[mask]
        column_sums = subset.sum(axis=0)
        summed_df = pd.DataFrame({'time section total emission': column_sums})
        dcgan = make_dcgan_gdf(summed_df, geoms)
        group_df = smell_reports_range.loc[start:end]
        if len(group_df) == 0:
            print('no data for this hour')
            continue
        else:
            cr_result_df = preprocess(group_df, bbox, crs)
            merged_avg = merge_average(cr_result_df, cell, time, plot=False)
            merged_sum = merge_sum(cr_result_df, cell, time, plot=False)
            pa_average, df_average = percentage_agreement(dcgan, merged_avg)
            pa_sum, df_sum = percentage_agreement(dcgan, merged_sum)
            
            df_add_to_list = df_average.copy()
            df_add_to_list = df_add_to_list.drop(columns=['geometry'])
            combined_df_average=  pd.concat([combined_df_average,df_add_to_list] , axis=0)
            
            impactstats_avg, corr_avg = get_impact_stats(df_average, time, plot=False, title='Average', plotsave=False)
            impactstats_sum, corr_sum = get_impact_stats(df_sum, time, plot=True, title='Sum', plotsave=True)
            
            month_results[time] = {
                'date': time,
                'percentage agreement average': pa_average,
                'percentage agreement sum': pa_sum,
                'impact stats average': impactstats_avg,
                'impact stats sum': impactstats_sum,
                'correlation average': corr_avg,
                'correlation sum': corr_sum} 
            
# UNCOMMENT TO SAVE THE RESULTS
# month_results_df = pd.DataFrame.from_dict(month_results, orient='index')
# save to csv, change the name of the file
# month_results_df.to_csv(f'stats/run{test_num}_11579_morningafternoon_{yr}{mo}.csv')

# PLOT
# imapct plot for the month just here
# get all the impact stats average out
impact_stats = month_results.get('impact stats average')
fig, ax = plt.subplots()
plt.plot(combined_df_average['smell value'], combined_df_average['emission'], 'rx', alpha=0.5)
means = combined_df_average.groupby('smell value').mean()
std = combined_df_average.groupby('smell value').std()
plt.plot(means.index, means['emission'])
plt.fill_between(std.index, means['emission']-std['emission'], means['emission']+std['emission'], alpha=0.2)
plt.savefig(f'images/impact_analysis_{yr}{mo}_average.pdf')
    
# This is a particular code that analyses a specified range of dates. 
ranges = [range(6, 14 + 1), range(14, 22 + 1)]
names = ['morning', 'afternoon']
years = [2022,2022,2022,2022,2022,2022,2022]
months = [2,3,6,9,10,11,12]
month_results = {}
pa_list = []
feb_daysofinterest = [1,7,15,16,22]
march_daysofinterest = [5,10,11,15,16,17]
june_daysofinterest = [30]
sept_daysofinterest = [16,17]
oct_daysofinterest = [11,12,14,21,22,23,25]
nov_daysofinterest = [2,4,10,22]
dec_daysofinterest = [5,29,30]
combined_df_average = pd.DataFrame()
combined_df_sum = pd.DataFrame()
all_daysofinterest = [feb_daysofinterest, march_daysofinterest, june_daysofinterest, sept_daysofinterest, oct_daysofinterest, nov_daysofinterest, dec_daysofinterest]

for yr, mo, daysofinterest in zip(years, months, all_daysofinterest):
    print(yr, mo)
    smell_reports_range = smell_reports[(smell_reports['Year']==yr) & (smell_reports['Month']==mo)]
    for day in daysofinterest:
        for name, hours in zip(names, ranges):
            time = f'{yr}-{mo}-{day}-{name}'
            print(time)
            start = pd.Timestamp(f'{yr}-{str(mo).zfill(2)}-{str(day).zfill(2)}-{str(hours[0]).zfill(2)}')
            end = pd.Timestamp(f'{yr}-{str(mo).zfill(2)}-{str(day).zfill(2)}-{str(hours[-1]).zfill(2)}')
            mask = (df.index >= start) & (df.index <= end)
            subset = df.loc[mask]
            column_sums = subset.sum(axis=0)
            summed_df = pd.DataFrame({'time section total emission': column_sums})
            dcgan = make_dcgan_gdf(summed_df, geoms)
            # get the cr reports in this time range
            group_df = smell_reports_range.loc[start:end]
            if len(group_df) == 0:
                print('no data for this hour')
                continue
            else:
                cr_result_df = preprocess(group_df, bbox, crs)
                merged_avg = merge_average(cr_result_df, cell, time, plot=False)
                merged_sum = merge_sum(cr_result_df, cell, time, plot=False)
                pa_average, df_average = percentage_agreement(dcgan, merged_avg)
                print('pa average', pa_average)
                pa_list.append(pa_average)
                pa_sum, df_sum = percentage_agreement(dcgan, merged_sum)
                print('pa sum', pa_sum)
                df_add_to_list = df_average.copy()
                df_add_to_list = df_add_to_list.drop(columns=['geometry'])
                combined_df_average=  pd.concat([combined_df_average,df_add_to_list] , axis=0)
                
                df_sum_add_to_list = df_sum.copy()
                df_sum_add_to_list = df_sum_add_to_list.drop(columns=['geometry'])
                combined_df_sum=  pd.concat([combined_df_sum,df_sum_add_to_list] , axis=0)
                    

# save these dfs, comment out if already done and then just load them in
combined_df_average.to_csv(f'combined_df_average_periodofimpact.csv')
combined_df_sum.to_csv(f'combined_df_sum_periodofimpact.csv')

# loading in code:
# combined_df_average = pd.read_csv('combined_df_average_periodofimpact.csv')
# combined_df_sum = pd.read_csv('combined_df_sum_periodofimpact.csv')

grouped = combined_df_average.groupby('smell value')
fig, ax = plt.subplots(1,1, figsize=(10,10))
hists = []
means =[]
smells = []
binwidth = 2
for name, group in grouped:
    smells.append(name)
    means.append(group['emission'].mean())
    ax.scatter(group['smell value'], group['emission'], label=name, color='blue', alpha=0.5)
ax.scatter(smells, means, color='red', marker='x')
ax.plot(smells, means, 'black')
plt.xlabel('Average smell value')
plt.ylabel('PM2.5 concentration ($\mu g/m^3$)')
plt.tight_layout()
plt.savefig(f'images/average_smellvalue_vs_emission_forperiodofinterest.pdf')

