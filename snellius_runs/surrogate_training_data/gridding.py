from datetime import datetime
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import shapely
from shapely.geometry import Point
import csv
import os, sys

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

def preprocess(df, hr, bbox_gdf, crs):
 coord_mapping = {
     '40.328015,-79.903551': 1, #Irvin
     '40.392967,-79.855709': 2, #ET
     '40.305062,-79.876692': 3, #Clairton
     '40.538261,-79.790391': 4, #Cheswick
     '40.479019,-79.960299': 5  #McConway
 }
 df['source'] = df['source'].map(coord_mapping)
 #  keep only the rows with correct hr
 df = df[df['HR']==hr]
 # Convert the coordinates to a Point geometry
 geometry = [Point(xy) for xy in zip(df['LON'], df['LAT'])]
 #  df = df.drop(['LON', 'LAT'], axis=1)
 # Create a GeoDataFrame using the original DataFrame and the geometry column
 gdf = gpd.GeoDataFrame(df, geometry=geometry)
 gdf = gdf.drop(['LON', 'LAT', 'YEAR', 'MO', 'DA','HR'], axis=1)
 gdf.crs = crs

 # choose only points within the boundary of the city
 points_in_bbox = gpd.sjoin(gdf, bbox_gdf, how="inner",predicate='within')
 joined_df = points_in_bbox.drop(columns=['index_right'])

 return joined_df



def merge(result_gdf, cell,scale=True, plot = False):
    merged = gpd.sjoin(result_gdf,cell, how='left', predicate='within')
    merged['total_emission']=1
    dissolve = merged.dissolve(by="index_right", aggfunc="sum")
    cell = cell.merge(dissolve['total_emission'], how='left', left_index=True, right_index=True)
    # print(cell)
    cell['total_emission'] = cell['total_emission'].fillna(0)
    if scale:
        min = 0.0
        max = 15.0
        cell['total_emission'] = (cell['total_emission'] - min) / (max - min)

    if plot:
        ax = cell.plot(column='total_emission', figsize=(12, 8), cmap='Blues', edgecolor="grey")

    return cell

def save(cell):
    cell['centroid_x'] = cell['geometry'].centroid.x
    cell['centroid_y'] = cell['geometry'].centroid.y

    # Drop the original geometry column
    gdf = cell.drop(columns=['geometry'])
    df = pd.DataFrame(gdf)
    return df


def find_polygon(cell, source_info):
    # find the grid where the source is located
    # dictionary of the source locations
    locations = {}
    for source in source_info.keys():
        x = source_info[source][1]
        y = source_info[source][0]
        point_geom = Point(x,y)
        # print(point_geom)
        contains_polygon = cell[cell.geometry.contains(point_geom)]
        # print(contains_polygon.index)
        locations[source] = contains_polygon.index
    return locations


def combine(df, input, locations):
    df['source'] = 0

    for source in locations.keys():
       index = locations[source]
       emission = input[source]
       df.loc[index, 'source'] = emission

    return df


def make_channels(gridsize, input, df, weather):
    """greating 9 channels for the input data with size of
    the gridsize of the map

    Args:
        gridsize (_type_): length of the grid
        input (_type_): weather parameters extracted from zrrra
        df (_type_): the converted geopandas dataframe containing hysplit results and source emission
        weather (array): list of weather parameters

    Returns:
        array: array of 9 channels
    """
    channels = []
    for param in weather:
        # channels[param] = np.full((gridsize,gridsize), input[param])
        channels.append(np.full((gridsize,gridsize), input[param]))
    print("weather channels made")
    emissions = np.array(df['total_emission']).reshape(gridsize,gridsize).T
    channels.append(emissions)
    # channels['emissions'] = emissions
    source = np.array(df['source']).reshape(gridsize,gridsize).T
    channels.append(source)
    # channels['source'] = source


    return channels

def make_output_channel(output_gridsize, df):
 """_summary_

 Args:
     gridsize (_type_): _description_
     df (_type_): the converted geopandas dataframe containing hysplit results and source emission

 Returns:
     _type_: _description_
 """
 emissions = np.array(df['total_emission']).reshape(output_gridsize,output_gridsize).T
 return emissions

def weather_load(df, time, variables, maxs, mins):
 weathers = []
 for var, max_val, min_val in zip(variables,maxs,mins):
  grid = df[var][time]
  # Removing newline characters from the string to form a continuous sequence
  data_str = grid.replace('\n', ' ')

  # Removing square brackets from the string to get a space-separated sequence
  data_str = data_str.replace('[', '').replace(']', '')

  # Convert the space-separated sequence to a numpy array
  numpy_array = np.fromstring(data_str, sep=' ')

  # this is the flattened array for the variable
  # this now needs to be minmax scaled
  scaled_array = (numpy_array - min_val) / (max_val - min_val)
  #weathers.append(scaled_array)
  weathers.append(scaled_array)
 array = np.array(weathers)

 return np.concatenate(array)

def generate_cyclical_encoding(val, period, start_num):
    sin_val = np.sin(2 * np.pi * (val-start_num) / period)
    cos_val = np.cos(2 * np.pi * (val-start_num) / period)
    return sin_val, cos_val

def time_encoding(timestr):
 time = []
 # make this into unix time
 dt_object = datetime.strptime(timestr, '%Y-%m-%d %H:%M:%S%z')
 # Get the Unix/Epoch time (timestamp) in seconds
 epoch_time = dt_object.timestamp()
 #  minmax scale the time (max = 1 April 2023, min = 1 august 2016)
 scaled_epochtime = (epoch_time - 1470002400) / (1680300000 - 1470002400)
 time.append(scaled_epochtime)

 vals = [dt_object.year, dt_object.month, dt_object.day, dt_object.hour]
 periods = [365, 12, 31, 24]
 start_nums = [0, 0, 1, 0]
 for val, period, start_num in zip(vals, periods, start_nums):
    sin, cos = generate_cyclical_encoding(val, period, start_num)
    time.append(sin)
    time.append(cos)

 return np.array(time)

def source_load(source_info, timestr):
 # load each source emission from file
 overall_min = -14
 overall_max = 356
 input = []
 for source in source_info.keys():
  print(source)
  # load the source emission
  source_df = pd.read_csv(f'/projects/0/gusr0543/surrogate/sensor_data/{source}_emission_filled-1.csv', index_col=0)
  source_data = source_df.rename(columns={'emission':source})
  # source_data['time'] = pd.to_datetime(source_data['time']).dt.strftime('%Y-%m-%d %H:%M:%S').astype('datetime64[ns]')
  value = source_data[source_data.index==timestr]
  print(value)
  input.append(value.values[0][0])
 # minmax scale the input
 input = np.array(input)
 scaled_source = (input - overall_min) / (overall_max - overall_min)

 return scaled_source


def main(argv):
    # some variables
    source_info = {'Glassport':(40.326009,-79.881703),
                'NorthBraddock':(40.402324,-79.860973),
                'Liberty':(40.323768,-79.868062),
                'Lawrenceville':(40.465420,-79.960757),
                'Harrison':(40.586372,-79.764863)}
    weather  = ['UGRD', 'VGRD', 'TMP', 'PRES', 'RH', 'SFCR', 'TCDC']
    maxs=[25,35, 315,1.043e+05, 1,100,100]
    mins=[-17,-19, 245, 8.485e+04, 0,0,0 ]
    gridsize = 64
    date = argv[1]
    year = date.split('_')[0]
    month = date.split('_')[1]

    # for PATH in folder:
    # load files
    weather_file = f'/projects/0/gusr0543/weather/input_data/weather_input_data_{year}_{month}.csv'
    # input_file = 'input_data_2020_8.csv'
    print("loading input file:", weather_file)
    # other input data
    weather_df = pd.read_csv(weather_file, index_col=0)


    for day in range(1,32):
        # check if file exists
        hysplit_file = f'/projects/0/gusr0543/zips/{year}-{str(month).zfill(2)}/run_{year}-{str(month).zfill(2)}-{str(day).zfill(2)}-00:00_24.0hr.csv'
        # hysplit_file = 'run_2019-01-16-00:00_24.0hr.csv'
        if os.path.exists(hysplit_file):
            print("loading hysplit results:", hysplit_file)
            result_raw = pd.read_csv(hysplit_file, skiprows=2)
            cell, bbox_gdf, crs = boundary_setup(gridsize)
            # for each hour do this:
            for hour in range(24):
                # get correct portion of hysplit file
                result_gdf = preprocess(result_raw, hour, bbox_gdf, crs)

                # get correct array from input data
                timestr = f'{year}-{str(month).zfill(2)}-{str(day).zfill(2)} {str(hour).zfill(2)}:00:00'
                # print("timestr type:", type(timestr))
                # print("timestr:", timestr)
                # input_row = input[input['time']==timestr]
                # print("input:", input_row)

                # input = []
                # scaled, flattened weather data
                weather_array = weather_load(weather_df, timestr, weather, maxs, mins)
                print(weather_array.shape)
                print("negatives in weather array? ",weather_array[weather_array<0].any())
                #input.append(weather_array)


                # grid the data up
                bare_grid = merge(result_gdf, cell)

                # Need a function to make a grid with source emissions (read from files), flatten and preprocess
                # add this flattened array to the weather data
                timestr = f'{year}-{str(month).zfill(2)}-{str(day).zfill(2)} {str(hour).zfill(2)}:00:00+00:00'
                source_array = source_load(source_info, timestr)
                print("source array",source_array.shape, source_array)
                #input.append(source_array)
                print("negatives in source array? ",source_array[source_array<0].any())
                # fianlly make a time flattened array and add to the rest
                time_array = time_encoding(timestr)
                #input.append(time_array)
                input = np.array([weather_array, source_array, time_array], dtype=object)
                #input = np.array(input)
                print("input pre flattening: ", input.shape, input)
                flat_input = np.concatenate(input)
                print("input post flattening: ", flat_input.shape, flat_input)
                print("Negatives in flat input? ", flat_input[flat_input<0].any())

                # find location of source in the grid
                # source_gridlocations = find_polygon(bare_grid, source_info)
                # add the source emission to the grid
                # grid_full = combine(bare_grid, input_row, source_gridlocations)
                # print("grid filled out with hysplit data and source emission")
                # final_data = make_flat_channels(gridsize, input_row, grid_full, weather)

                # emissions only
                output = make_output_channel(gridsize, bare_grid)
                print("output channel made", output.shape)

                # save to one csv, so when loading you can just index the first row and then then rest
                with open(f'training_data/training_{year}_{month}_{day}_{hour}.csv', mode='w', newline='') as file:
                 writer = csv.writer(file)
                 writer.writerow(flat_input)
                 writer.writerows(output)

                print("saved to file:", f'training_{year}_{month}_{day}_{hour}.csv')

        else:
            # make the input files for the missing days so that they can be generated
            print("file does not exist:", hysplit_file)
            print("making input file to be used to generate with DCGAN")
            for hour in range(24):
                timestr = f'{year}-{str(month).zfill(2)}-{str(day).zfill(2)} {str(hour).zfill(2)}:00:00'
                # scaled, flattened weather data
                weather_array = weather_load(weather_df, timestr, weather, maxs, mins)
                # Need a function to make a grid with source emissions (read from files), flatten and preprocess
                # add this flattened array to the weather data
                timestr = f'{year}-{str(month).zfill(2)}-{str(day).zfill(2)} {str(hour).zfill(2)}:00:00+00:00'
                source_array = source_load(source_info, timestr)
                print(source_array)
                # fianlly make a time flattened array and add to the rest
                time_array = time_encoding(timestr)
                # input = np.array(input)
                input = np.array([weather_array, source_array, time_array], dtype=object)
                print("input pre flattening: ", input.shape, input)
                flat_input = np.concatenate(input)
                print("input post flattening: ", flat_input.shape, flat_input)

                # save to one csv, so when loading you can just index the first row and then then rest
                with open(f'input_data/input_{year}_{month}_{day}_{hour}.csv', mode='w', newline='') as file:
                 writer = csv.writer(file)
                 writer.writerow(flat_input)

                print("saved to file:", f'input_{year}_{month}_{day}_{hour}.csv')



    print("DONE!!!!!!!")


if __name__ == "__main__":
    main(sys.argv)


