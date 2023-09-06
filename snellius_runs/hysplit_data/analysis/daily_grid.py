
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
 # Convert the coordinates to a Point geometry
 geometry = [Point(xy) for xy in zip(df['LON'], df['LAT'])]
 #  df = df.drop(['LON', 'LAT'], axis=1)
 # Create a GeoDataFrame using the original DataFrame and the geometry column
 gdf = gpd.GeoDataFrame(df, geometry=geometry)
 gdf = gdf.drop(['LON', 'LAT'], axis=1)
 gdf.crs = crs

 # choose only points within the boundary of the city
 points_in_bbox = gpd.sjoin(gdf, bbox_gdf, how="inner",predicate='within')
 joined_df = points_in_bbox.drop(columns=['index_right'])

 return joined_df



def merge(result_gdf, cell,date, day,plot = False, scale = False, save = False):
    merged = gpd.sjoin(result_gdf, cell, how='left', predicate='within')

    cell_max_smell = merged.groupby('index_right')['TEST'].sum()
    cell = cell.merge(cell_max_smell, how='left', left_index=True, right_index=True)
    cell['TEST'] = cell['TEST'].fillna(0)
    if scale:
        min = 0.0
        max = 15.0
        cell['TEST'] = (cell['TEST'] - min) / (max - min)
    if plot:
        ax = cell.plot(column='TEST', figsize=(12, 8), cmap='BuGn', edgecolor="white")
        # ax.set_axis_off()
        # ax.colorbar()
        sm = plt.cm.ScalarMappable(cmap='BuGn')
        sm.set_array(cell['TEST'])
        # norm = mcolors.Normalize(vmin=0, vmax=5)
        # sm.set_norm(norm)
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Emission')  # Add a label to the colorbar
    if save == True:
     cell.to_file(f'only_output/{date}-{day}.geojson', driver='GeoJSON')
     print(f"file for {date}-{day} saved")
    return cell



def main(argv):
 gridsize = 64
 date = argv[1]
 year = date.split('-')[0]
 month = date.split('-')[1]
 for day in range(1,32):
  hysplit_file = f'/projects/0/gusr0543/zips/{year}-{str(month).zfill(2)}/run_{year}-{str(month).zfill(2)}-{str(day).zfill(2)}-00:00_24.0hr.csv'
  if os.path.exists(hysplit_file):
    print(f'Processing {hysplit_file}')
    day_result = pd.read_csv(hysplit_file, skiprows=1, usecols=["LON", "LAT", "TEST"])
    cell, bbox, crs = boundary_setup(gridsize)
    result_gdf = preprocess(day_result, bbox, crs)
    merged = merge(result_gdf, cell, date,day, plot=False,scale = True , save = True)
    #print(f"max: {merged['TEST'].max()} and min: {merged['TEST'].min()}")

  else:
      print("file doesn't exist:", hysplit_file)

 print("________DONE_________")
if __name__ == "__main__":
    main(sys.argv)
