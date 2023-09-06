import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx
# define which basemap to use
basemap = ctx.providers.CartoDB.Positron


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
 df = df.drop(['LON', 'LAT'], axis=1)

 # Create a GeoDataFrame using the original DataFrame and the geometry column
 gdf = gpd.GeoDataFrame(df, geometry=geometry)
 gdf = gdf.set_crs(epsg=4269)
 # df['coordinate'] = df['LAT'].astype(str) + ',' + df['LON'].astype(str)
 return gdf

def plot_forhour(basemap, df, hour):
 # basemap = ctx.providers.OpenStreetMap.Mapnik
 df = df[df['HR']==hour]
 df = df.set_crs(epsg=4269)
 xmin, xmax = 40.2393, 40.5746
 ymin, ymax = -80.1676, -79.7231

 fig, ax = plt.subplots(figsize=(10, 10))
 ax.set_xlim(xmin, xmax)
 ax.set_ylim(ymin, ymax)
 df.plot(ax=ax, markersize=4, alpha=df['TEST']/df['TEST'].max(),column='source', cmap = 'Set1')
 crs = 'EPSG:4326'
 #ctx.add_basemap(ax, crs=crs, source = basemap)

 # ax.set_axis_off()
 # xmin, xmax = 40.2393, 40.5746
 # ymin, ymax = -80.1676, -79.7231
 # ax.set_xlim(xmin, xmax)
 # ax.set_ylim(ymin, ymax)
 ctx.add_basemap(ax, crs=crs, source = basemap)
 plt.savefig("test.png")

test1 = pd.read_csv('2021-04/run_2021-04-10-00:00_24.0hr.csv', skiprows=2)
test1_gdf = preprocess(test1)
plot_forhour(basemap, test1_gdf, 15)

