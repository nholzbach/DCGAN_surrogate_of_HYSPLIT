import swifter
import numba
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
# https://contextily.readthedocs.io/en/latest/places_guide.html

# preprocessing a little bit
# colour by source
def preprocess(df):
 coord_mapping = {
     '40.328015,-79.903551': 1, #Irvin
     '40.392967,-79.855709': 2, #ET
     '40.305062,-79.876692': 3, #Clairton
     '40.538261,-79.790391': 4, #Cheswick
     '40.479019,-79.960299': 5  #McConway
 }
 df['source'] = df['source'].map(coord_mapping)

 # Convert the coordinates to a Point geometry
 geometry = [Point(xy) for xy in zip(df['LON'], df['LAT'])]
 #  df = df.drop(['LON', 'LAT'], axis=1)
 # Create a GeoDataFrame using the original DataFrame and the geometry column
 gdf = gpd.GeoDataFrame(df, geometry=geometry)

 # df['coordinate'] = df['LAT'].astype(str) + ',' + df['LON'].astype(str)
 return gdf

def build_monitoring_station_gdf():
    columns = ['name', 'lat', 'lon', 'data_root']
    monitoring_stations = pd.DataFrame(columns=columns)
    stations = [
        {'name':'Liberty', 'lat':40.323768, 'lon':-79.868062, 'data_root':'/projects/0/gusr0543/zips/sensor_data/Liberty_emission_filled-1.csv'},
        {'name': 'Glassport', 'lat':40.326009, 'lon':-79.881703, 'data_root':'/projects/0/gusr0543/zips/sensor_data/Glassport_emission_filled-1.csv'},
        {'name': 'Harrison', 'lat':40.617488, 'lon':-79.727664, 'data_root': '/projects/0/gusr0543/zips/sensor_data/Harrison_emission_filled-1.csv'},
        {'name': 'NorthBraddock', 'lat':40.402324, 'lon':-79.860973, 'data_root':'/projects/0/gusr0543/zips/sensor_data/NorthBraddock_emission_filled-1.csv'},
        {'name':'Lawrenceville', 'lat':40.465420, 'lon':-79.960757, 'data_root': '/projects/0/gusr0543/zips/sensor_data/Lawrenceville_emission_filled-1.csv' }
    ]
    # Iterate over the sources and extract the attributes
    # make a df from the stations dict
    monitoring_stations = pd.DataFrame(stations, columns=columns)
    # Create the GeoDataFrame
    # Convert lat and lon columns to numeric types
    monitoring_stations['lat'] = monitoring_stations['lat'].astype(float)
    monitoring_stations['lon'] = monitoring_stations['lon'].astype(float)


    # Create a Point geometry column from lat and lon columns
    monitoring_stations_gdf = gpd.GeoDataFrame(monitoring_stations, geometry=gpd.points_from_xy(monitoring_stations['lon'], monitoring_stations['lat']))
    # Set the CRS of the GeoDataFrame
    monitoring_stations_gdf.crs = 'EPSG:4326'

    # Print the GeoDataFrame
    print(monitoring_stations_gdf)
    return monitoring_stations_gdf

def preprocess_monitor(df):
    df.index = pd.to_datetime(df.index)
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day
    df['Hour'] = df.index.hour
    return df

def matching_date(df, year, month, date):
    matching = df[(df['Year'] == year) & (df['Month'] == month) & (df['Day'] == date)]
    return matching

def within_radius(row, coord, search_radius):
    """filter instances based on their distance from the center point (sensor)
    """
    point = (row['LAT'], row['LON'])
    return distance.distance(coord, point).meters <= search_radius

# Draxler ranking method: (for different methods, so can compare this to ML method)
# correlation coefficient R, fractional bias (FB), figure of merit in space (FMS), and Kolmogorov–Smirnov parameter (KSP)
# Rank = R2 + 1 − |F B/2| + F M S/100 + (1 − KSP/100

def correlation_coeff(predictions, targets):
    # this is pearson correlation, what hysplit paper uses
    return pearsonr(predictions, targets)

def RANK(cor, fb):
    return cor**2 + 1 - np.abs(fb/2)
# + fms/100 + (1-ksp/100)

def FB(predictions, targets):
    """ Fractional Bias """
    return 2 * (np.mean(predictions) - np.mean(targets))/(np.mean(predictions) + np.mean(targets))

def KSP(predictions, targets):
    """ Kolomogorov–Smirnov """
    # actually not sure if this is appropriate
    return ks_2samp(predictions, targets)

# Simple rmse function
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def get_rmse_correlation(station_df, station_name, yr, month, day, dict_of_results, zoom=1000 ):
    """Generates results analysis for a given station and data.
    Returns:
    RMSE value
    Comparison plot
    Correlation coef R and its p value

    Args:
        station_df (df): the raw dataframe of the station data (not preprocessed)
        station_name (str): the name of the station, format used in the dict_of_results
        date (tuple): (yr, month, day) of interest
    """
    # get the predictions (sum of all readings in the hour at points within 500m from station)
    results = {}

    predictions = dict_of_results[station_name].groupby('HR')['TEST'].sum()
    predictions = predictions.to_frame()
    full_day = np.arange(0,24,1)
    predictions = predictions.reindex(full_day, fill_value=0)

    # get the targets (matching date from the station data)
    station = preprocess_monitor(station_df)
    matching_station = matching_date(station, yr, month, day)
    targets =np.array( matching_station['emission'])

    predictions = np.array(predictions).reshape(targets.shape)
    # return the rmse
    rmse_result = rmse(predictions, targets)
    results['rmse'] = rmse_result
    print("RMSE: ", rmse_result)
    # fig,ax = plt.subplots()
    # ax.plot(predictions*zoom, label=f'predictionsx{zoom}')
    #ax.plot(targets, label='targets')
    #ax.legend()
    #ax.set_xlabel("Hour of the Day")
    #ax.set_title(station_name)



    # now for the correlation resutls
    r, p  = correlation_coeff(predictions, targets)
    results['r'] = r
    results['p'] = p
    results['predictions'] = predictions
    results['targets'] = targets
    print(f"correlation coeff {r}, p-value {p}")

    # df = pd.DataFrame(results)
    # print(f"saving analysis results for {station_name} on {yr}-{month}-{day} to csv")
    # df.to_csv(f'results/{station_name}_{yr}_{month}_{day}.csv')
    # return predictions, targets, rmse_result, r, p
    return results

def rmse_skip_neg(station_df, station_name, yr, month, day, dict_of_results):
    results = {}
    instances_within_radius = dict_of_results[station_name]
    if instances_within_radius.empty:
        print("No instances found for this station, so it's being skipped")
    else:
        predictions = instances_within_radius.groupby('HR')['TEST'].sum()
        predictions = predictions.to_frame()
        full_day = np.arange(0,24,1)
        predictions = predictions.reindex(full_day, fill_value=0)
        print("predictions:",predictions)
        # get the targets (matching date from the station data)
        station = preprocess_monitor(station_df)
        matching_station = matching_date(station, yr, month, day)
        #print(matching_station['emission'][0])
        if matching_station['emission'][0]==-1:
            print("No matching data in mon station for this day", yr, month, day)
            return None
        else:
            targets = matching_station['emission'].to_frame()
            targets.index = matching_station['Hour']
            print("targets:", targets)
            combined = pd.concat([predictions, targets], axis=1)
            df_filtered = combined[combined['emission'] >= 0]
            predictions = np.array(df_filtered['TEST'])
            targets = np.array(df_filtered['emission'])
            # rmse = np.sqrt(((df_filtered['TEST'] - df_filtered['emission']) ** 2).mean())
            rmse_result = rmse(predictions, targets)
            results['rmse'] = rmse_result
            print("RMSE: ", rmse_result)

            # correlation resutls
            r,p = correlation_coeff(predictions, targets)
            results['r'] = r
            results['p'] = p
            results['predictions'] = predictions
            results['targets'] = targets
            print(f"correlation coeff {r}, p-value {p}")

            return results

def process_run(run_results, PATH, monitoring_stations_gdf, save_instances=True, search_radius=500):
    """Steps:   1. Load run results and preprocess
                2. Iterate over stations and find instances within radius
                        (this requires the monitoring_stations_gdf to be defined previously)
                3. Save these instances to csv for playing around with later
                4. Do rmse analysis and plot for each station
                    This can be removed if we don't want to do this for every run, but it's nice
                    to have to check that results are reasonable
                    (it's in this function that the station data is loaded, preprocessed and matched to date)

    Args:
        run_results (str): the name of the file of hysplit results
        search_radius (int, optional): _description_. Defaults to 500.
    """
    # get date out
    date = run_results.split('_')[1]
    yr = int(date.split('-')[0])

    # get month but if 0 is before, discard
    month = date.split('-')[1]
    # remove 0 from the front and make it an int
    month = int(month.lstrip("0"))
    day = date.split('-')[2]
    # same here
    day = int(day.lstrip("0"))
    # print(day)

    # read csv
    #PATH = 'HYSPLIT_results/'
    result_raw = pd.read_csv(PATH+run_results, skiprows=2)
    result_gdf = preprocess(result_raw)
    # print(result_gdf.head())
    print("loaded and preprocessed file:", run_results)
    zooms = [1000,1000,10, 100000,100000]
    dict_of_results = {}
    analysis_results = {}
    for i, zoom in zip(np.arange(0,5,1), zooms):
        station_name =monitoring_stations_gdf.iloc[i]['name']
        print("finding instances within radius for station: ", station_name)

        # load raw data
        # if i build this into another function that runs this function mulitple times,
        # remove this out of this loop so that it's only loaded once
        # print(monitoring_stations_gdf.iloc[i]['data_root'])
        raw_data = pd.read_csv(monitoring_stations_gdf.iloc[i]['data_root'], index_col=0)
        monitor_coord = (monitoring_stations_gdf.iloc[i]['lat'],monitoring_stations_gdf.iloc[i]['lon'])
        # filter instances within radius
        filename = f'/projects/0/gusr0543/zips/instances/{station_name}_instances_{yr}_{month}_{day}.csv'
        print(filename)
        if not os.path.exists(filename):
            # if save_instances==True:
            print("Generating this info")
            instances_within_radius = result_gdf[result_gdf.swifter.apply(within_radius, axis=1, coord=monitor_coord, search_radius=search_radius)]
            dict_of_results[station_name] = instances_within_radius
            # also save this because it takes so long to run
            # dict_of_results[station_name].to_csv(f'/projects/0/gusr0543/zips/instances/{station_name}_instances_{yr}_{month}_{day}.csv')
            dict_of_results[station_name].to_csv(filename)
        else:
            print("Loading this info from csv")
            instances_within_radius = pd.read_csv(filename)
            # instances_within_radius = pd.read_csv(f'/projects/0/gusr0543/zips/instances/{station_name}_instances_{yr}_{month}_{day}.csv')
            dict_of_results[station_name] = instances_within_radius
            if instances_within_radius.empty:
                print("There are no instances found within the radius for this station")

        # do actual rmse analysis and plot
        print("Now doing rmse analysis and plot")
        # station_result = get_rmse_correlation(raw_data, station_name, yr, month, day, dict_of_results, zoom)
        station_result = rmse_skip_neg(raw_data, station_name, yr,month, day, dict_of_results)
        analysis_results[station_name] = station_result
    date_df = pd.DataFrame(analysis_results)
    date_df.to_csv(f'results/{yr}_{month}_{day}_negativetest.csv')


    # return predictions, targets, rmse_result, r, p


def main(argv):
    month = argv[1]
    PATH = "/projects/0/gusr0543/zips/"+month+"/"
    monitoring_stations_gdf = build_monitoring_station_gdf()
    print("monitoring stations loaded")
    for file in os.listdir(PATH):
        print("working on:", file)
        process_run(file, PATH ,monitoring_stations_gdf, save_instances=False)
    print("All done!!!!!")

if __name__ == "__main__":
    main(sys.argv)
