
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
sns.set_style("whitegrid")
sns.set_context("talk", font_scale=1.5)
import shapely
from shapely.geometry import Point

def plot_grids(hysplit_grid, cr_grid, date, save=False):
 fig , ax = plt.subplots(1,2, figsize = (10,10))

 # plot the two grids
 ax[0] = hysplit_grid.plot(column='TEST', alpha=0.5, ax = ax[0], cmap = 'rocket_r')
 ax[0].set_title('HYSPLIT emissions results')
 sm = plt.cm.ScalarMappable(cmap = 'rocket_r')
 sm.set_array(hysplit_grid['TEST'])
 cbar = plt.colorbar(sm, ax=ax[0], shrink=0.4)

 ax[1]= cr_grid.plot(column='smell value', ax=ax[1], alpha=0.5, cmap = 'rocket_r')
 ax[1].set_title('Citizen reports')
 sm2 = plt.cm.ScalarMappable(cmap = 'rocket_r')
 sm2.set_array(cr_grid['smell value'])
 cbar = plt.colorbar(sm2, ax=ax[1], shrink=0.4)
 if save == True:
  fig.tight_layout()
  fig.savefig(f'images/grid_comparison_{date}.pdf')


def get_impact_stats(df, date, plot=False,title=None, save=False):
 df.drop(columns=['geometry'], inplace=True)

 means = df.groupby('smell value').mean()
 means.rename(columns={'TEST': 'Mean'}, inplace=True)

 std = df.groupby('smell value').std()
 std.rename(columns={'TEST': 'STD'}, inplace=True)

 counts = df.groupby('smell value').count()
 counts.rename(columns={'TEST': 'Count'}, inplace=True)

 corr = df.corr()

 stats = pd.concat([means, counts, std], axis=1)

 if plot == True:
  fig, ax = plt.subplots(figsize=(10,10))
  plt.plot(df['smell value'], df['TEST'], 'rx', alpha=0.5, markersize=20,markeredgewidth=6)
  plt.plot(stats.index, stats['Mean'], linewidth=6)
  plt.fill_between(stats.index, stats['Mean']-stats['STD'], stats['Mean']+stats['STD'], alpha=0.2)
  if title == 'avg':
      plt.xlabel('Average smell value')
  elif title == 'sum':
      plt.xlabel('Summed smell value')
  plt.ylabel('PM2.5 concentration $\mu g/m^3$')
  if save == True:
    plt.tight_layout()
    plt.savefig(f'images/impact_analysis_{date}_{title}.pdf')

 return stats, corr['TEST']['smell value']


def percentage_agreement(hysplit, cr):
    # remove 0 values and geometry column
    # convert 0 to na
    cr['smell value'] = cr['smell value'].replace(0, np.nan)
    df1 = cr.dropna()
    print("cr no 0", type(df1), df1)
    print("hysplit geodf",type(hysplit), hysplit)
    if len(df1)==0:
        print('nothing in CR')
        return 0, None

    # Merge the two DataFrames based on their geometries
    merged_df = df1.merge(hysplit, on='geometry', suffixes=('_df1', '_df2'))

    # Count the number of matching geometries
    nonzero_vals = len(merged_df.where(merged_df['TEST'] > 0).dropna())

    # Calculate the percentage agreement
    total_values = len(df1)
    pa = (nonzero_vals / total_values) * 100
    return pa, merged_df


def boundary_setup(n_cells):
    # X = lon, Y = lat
    xmin, ymin, xmax, ymax = -80.1892, 40.2246, -79.6912, 40.5963
    # how many cells across and down
    cell_size_x = (xmax - xmin) / n_cells
    cell_size_y = (ymax - ymin) / n_cells
    # projection of the grid
    crs = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs"
    # create the cells in a loop
    grid_cells = []
    for x0 in np.arange(xmin, xmax, cell_size_x):
        for y0 in np.arange(ymin, ymax, cell_size_y):
            # bounds
            x1 = x0 + cell_size_x
            y1 = y0 + cell_size_y
            grid_cells.append(shapely.geometry.box(x0, y0, x1, y1))
    cell = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=crs)

    # create shape for the bounding box
    bbox = shapely.geometry.box(xmin, ymin, xmax, ymax)
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs=crs)
    return cell, bbox_gdf, crs

def preprocess(df, bbox_gdf, crs, cols=['LON', 'LAT']):
 # Convert the coordinates to a Point geometry
 geometry = [Point(xy) for xy in zip(df[cols[0]], df[cols[1]])]
 # Create a GeoDataFrame using the original DataFrame and the geometry column
 gdf = gpd.GeoDataFrame(df, geometry=geometry)
 gdf = gdf.drop([cols[0], cols[1]], axis=1)
 gdf.crs = crs
 # choose only points within the boundary of the city
 points_in_bbox = gpd.sjoin(gdf, bbox_gdf, how="inner",predicate='within')
 joined_df = points_in_bbox.drop(columns=['index_right'])

 return joined_df

# for CR summing
def merge_sum(result_gdf, cell,date,group=None, plot = False):
    merged = gpd.sjoin(result_gdf, cell, how='left', predicate='within')
    # print(merged)
    cell_max_smell = merged.groupby('index_right')['smell value'].sum()
    # print(cell_max_smell)
    cell = cell.merge(cell_max_smell, how='left', left_index=True, right_index=True)
    cell['smell value'] = cell['smell value'].fillna(0)
    if plot:
        ax = cell.plot(column='smell value', figsize=(12, 8), cmap='BuGn', edgecolor="white")
        sm = plt.cm.ScalarMappable(cmap='BuGn')
        sm.set_array(cell['smell value'])

        norm = mcolors.Normalize()
        sm.set_norm(norm)
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Smell Value')  # Add a label to the colorbar
        ax.set_title(f'Summed smell value for {date}')
    return cell

# for CR averaging
def merge_average(result_gdf, cell,date, group=None, plot = False):
    merged = gpd.sjoin(result_gdf, cell, how='left', predicate='within')
    # print(merged)
    cell_max_smell = merged.groupby('index_right')['smell value'].mean()
    # print(cell_max_smell)
    cell = cell.merge(cell_max_smell, how='left', left_index=True, right_index=True)
    cell['smell value'] = cell['smell value'].fillna(0)
    if plot:
        ax = cell.plot(column='smell value', figsize=(12, 8), cmap='BuGn', edgecolor="white")
        sm = plt.cm.ScalarMappable(cmap='BuGn')
        sm.set_array(cell['smell value'])

        norm = mcolors.Normalize(vmin=0, vmax=5)
        sm.set_norm(norm)
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Smell Value')  # Add a label to the colorbar
        ax.set_title(f'Average smell value for {date}')
    return cell

# for HYSPLIT files
def merge(result_gdf, cell,date, plot = False, save = False):
    merged = gpd.sjoin(result_gdf, cell, how='left', predicate='within')

    cell_max_smell = merged.groupby('index_right')['TEST'].sum()
    cell = cell.merge(cell_max_smell, how='left', left_index=True, right_index=True)
    cell['TEST'] = cell['TEST'].fillna(0)
    if plot:
        ax = cell.plot(column='TEST', figsize=(12, 8), cmap='BuGn', edgecolor="white")
        sm = plt.cm.ScalarMappable(cmap='BuGn')
        sm.set_array(cell['TEST'])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Emission')  # Add a label to the colorbar

    if save == True:
     cell.to_file(f'only_output/{date}.geojson', driver='GeoJSON')
    return cell



def hysplit_to_grid(file):
    # read in hysplit results
    df = pd.read_csv(file, header=None, sep='\s+')
    df.columns = ['lat', 'lon', 'TEST']
    # make a geodataframe
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
    return gdf

def main(argv):
  yr, month = argv[1], argv[2]
  gridsize = 64
  cell, bbox, crs = boundary_setup(gridsize)
  #  adjust this path
  smell_reports = pd.read_csv('/projects/0/gusr0543/citizen_reports/raw/smell_reports.csv', usecols=["epoch time","skewed latitude", "skewed longitude","smell value"]).set_index("epoch time")
  smell_reports.index = pd.to_datetime(smell_reports.index, unit='s')
  smell_reports['Year'] = smell_reports.index.year
  smell_reports['Month'] = smell_reports.index.month
  smell_reports['Day'] = smell_reports.index.day
  smell_reports['Hour'] = smell_reports.index.hour

  stats = {}
  merged_df_list = []
  for day in range(1,32):
    hysplit_file =  f'/projects/0/gusr0543/zips/{yr}-{str(month).zfill(2)}/run_{yr}-{str(month).zfill(2)}-{str(day).zfill(2)}-00:00_24.0hr.csv'
    if os.path.exists(hysplit_file):
      print(f'Processing {hysplit_file}')
      # also need to do this by hour/chunk, not by day
      hysplit_day_result = pd.read_csv(hysplit_file, skiprows=1, usecols=["LON", "LAT", "TEST", "HR"])
      for hour in range(0,24):
            time = f'{yr}-{month}-{day}-{hour}'
            print("working on:", time)
            # get all the results for that hour
            hysplit_hour_result = hysplit_day_result.loc[hysplit_day_result['HR'] == hour]
            hysplit_result_gdf = preprocess(hysplit_hour_result, bbox, crs)
            hysplit_merged = merge(hysplit_result_gdf, cell, time, plot=False, save = False)


            # read in cr grid
            # match the hysplit date all the indexes to find in cr grid
            group_df = smell_reports.loc[(smell_reports['Year'] == int(yr)) & (smell_reports['Month'] == int(month)) & (smell_reports['Day'] == day) & (smell_reports['Hour'] == hour)]
            print(group_df)
            if len(group_df) == 0:
                print('no reports for this hour')
                continue

            else:

                cr_result_df = preprocess(group_df, bbox, crs, ['skewed longitude', 'skewed latitude'])
                cr_merged_avg = merge_average(cr_result_df, cell, time, plot=False)
                cr_merged_sum = merge_sum(cr_result_df, cell, time, plot=False)

                pa_avg, df_avg = percentage_agreement(hysplit_merged, cr_merged_avg)
                pa_sum, df_sum = percentage_agreement(hysplit_merged, cr_merged_sum)

                if (df_avg is not None) or (df_sum is not None):
                    print("making images")
                    impact_stats_avg, corr_avg = get_impact_stats(df_avg, time, plot=True, title="avg",save=True)
                    impact_stats_sum, corr_sum = get_impact_stats(df_sum, time, plot=True, title='sum',save=True)
                else:
                    impact_stats_avg, corr_avg = None, None
                    impact_stats_sum, corr_sum = None, None

if __name__ == '__main__':
    main(sys.argv)


