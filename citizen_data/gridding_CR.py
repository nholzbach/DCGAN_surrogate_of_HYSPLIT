import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import shapely
from shapely.geometry import Point
import os, sys
import csv
from datetime import datetime
import matplotlib.colors as mcolors

def boundary_setup(n_cells):
    """ Setup the grid system for the city of Pittsburgh with geoPandas
    returns:
    cell - the grid system
    bbox_gdf - the boundary of the city
    crs - the coordinate reference system
    """
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

def preprocess(df, bbox_gdf, crs):
    """
    Preprocess the citizen report data
    returns:
    joined_df - the citizen report data with the geometry column"""
 # Convert the coordinates to a Point geometry
 geometry = [Point(xy) for xy in zip(df['skewed longitude'], df['skewed latitude'])]
 # Create a GeoDataFrame using the original DataFrame and the geometry column
 gdf = gpd.GeoDataFrame(df, geometry=geometry)
 gdf = gdf.drop(['skewed longitude', 'skewed latitude'], axis=1)
 gdf.crs = crs

 # choose only points within the boundary of the city
 points_in_bbox = gpd.sjoin(gdf, bbox_gdf, how="inner",predicate='within')
 joined_df = points_in_bbox.drop(columns=['index_right'])
 return joined_df



def merge_sum(result_gdf, cell,date,group=None, plot = False, save = False):
    """ Merge the citizen report data with the grid system. Sum all the smell values occuring in the same cell
    input:
    result_gdf - the citizen report data with the geometry column
    cell - the grid system
    date - the date of the data
    group - the group of data in terms of time resolution (morning, afternoon, or none for hourly)
    """
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
        ax.set_title(f'Summed smell value for {group} on {date}')
    
    if save:
     if group == None:
         cell.to_file(f'grids/sum/{date}.geojson', driver='GeoJSON')
     else:
         cell.to_file(f'grids/sum/{date}-{group}.geojson', driver='GeoJSON')
    
    return cell

def merge_average(result_gdf, cell,date, group=None, plot = False, save = False):
    """ Merge the citizen report data with the grid system. Average all the smell values occuring in the same cell
    input:
    result_gdf - the citizen report data with the geometry column
    cell - the grid system
    date - the date of the data
    group - the group of data in terms of time resolution (morning, afternoon, or none for hourly)
    """
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
        cbar.set_label('Smell Value')  # Add a label to the colorbar
        ax.set_title(f'Average smell value for {group} on {date}')
    
    if save:
        if group == None:
            cell.to_file(f'grids/average/{date}.geojson', driver='GeoJSON')
        else:
            cell.to_file(f'grids/average/{date}-{group}.geojson', driver='GeoJSON')
    
    return cell


def save(cell):
    
    cell['centroid_x'] = cell['geometry'].centroid.x
    cell['centroid_y'] = cell['geometry'].centroid.y

    # Drop the original geometry column
    gdf = cell.drop(columns=['geometry'])
    df = pd.DataFrame(gdf)
    return df


def main(args):
    """ Main function to run the script. The script grids the smell reports and saves the grids as geojson files.
    3 time resolutions are done: hourly, morning/afternoon, and daily.
    Adjust save to True to save the grids as geojson files.
    
    The following directories must be created before running the script:
    - grids/sum
    - grids/average
    
    
    Args:
        args (str): The date to run the script on
    """
    
    smell_reports = pd.read_csv('raw/smell_reports.csv', usecols=["epoch time","skewed latitude", "skewed longitude","smell value"]).set_index("epoch time")
    smell_reports.index = pd.to_datetime(smell_reports.index, unit='s')
    
    morning = []
    afternoon = []
    night = [] 
    
    # grouped at daily resolution
    cell, bbox, crs = boundary_setup(64)
    grouped_by_day = smell_reports.groupby(smell_reports.index.date)
    for day, rows in grouped_by_day:
    # fig,ax = plt.subplots()
    if str(day) == args:
        print(f"Rows for {day}:")
        day_df = smell_reports.loc[str(day)]
        result_gdf = preprocess(day_df, bbox, crs) 
        merged = merge(result_gdf, cell, day, plot=False, save = True)
    
    # grouped at hourly resolution
    grouped_by_hour = smell_reports.groupby(smell_reports.index.floor('H'))
    for dayhour, rows in grouped_by_hour:
        # print(dayhour)
        if str(dayhour.date()) == args:
            group_df = smell_reports.loc[rows.index]
            result_gdf = preprocess(group_df, bbox, crs)
            merged_avg = merge_average(result_gdf, cell, dayhour, plot=False, save = True)
            merged_sum = merge_sum(result_gdf, cell, dayhour, plot=False, save = True)
 
        # group at morning/afternoon resolution
            if dayhour.hour >= 6 and dayhour.hour < 14:
                morning.append(rows)
            elif dayhour.hour >= 14 and dayhour.hour < 22:
                afternoon.append(rows)
            elif dayhour.hour >= 22 or dayhour.hour < 6:
                night.append(rows)

    for group, name in zip([morning, afternoon],['morning', 'afternoon']):
        group_df = pd.concat(group)
        
        result_gdf = preprocess(group_df, bbox, crs)
        merged_avg = merge_average(result_gdf, cell, day, name, plot=True, save = True)
        merged_sum = merge_sum(result_gdf, cell, day, name, plot=True, save = True)

if __name__ == "__main__":
    main(sys.argv)

