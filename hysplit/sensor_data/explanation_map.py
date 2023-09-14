from geopy.geocoders import Nominatim
import folium
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib.lines import Line2D
geolocator = Nominatim(user_agent="my_map")
location = geolocator.geocode("Pittsburgh")
latitude = location.latitude
longitude = location.longitude

# Create a map centered at the first location
map_object = folium.Map(location=[latitude, longitude], zoom_start=12)

# Save the map to an HTML file
# map_object.save("map.html")

sources = ["Irvin", "ET", "Clairton", "Cheswick", "McConway"]
source_lats = [40.328015,40.392967,40.305062,40.538261,40.479019]
source_lons = [-79.903551,-79.855709,-79.876692, -79.79039, -79.960299]
stations = ["Liberty","Glassport","Harrison", "North Braddock", "Lawrenceville"]
station_lats =[40.323768,40.326009,40.617488,40.402324, 40.465420]
station_lons =[-79.868062, -79.881703, -79.727664, -79.860973, -79.960757]

for i in range(len(source_lats)):
# Set 1 markers with one appearance
 folium.Marker(
     location=[source_lats[i], source_lons[i]],
     icon=folium.Icon(color="red", icon="cloud"),
     tooltip="Marker 1"
 ).add_to(map_object)

# Set 2 markers with a different appearance
 folium.Marker(
     location=[station_lats[i], station_lons[i]],
     icon=folium.Icon(color="blue", icon="info sign")
 ).add_to(map_object)

# map_object.save("map.html")

basemap = ctx.providers.CartoDB.Positron
fig, ax = plt.subplots(figsize=(10, 10))

# Plot markers on the axis
for i in range(len(source_lats)):
 ax.plot(source_lons[i], source_lats[i], 'r*', markersize=10, label = sources[i])

 ax.plot(station_lons[i], station_lats[i], 'bo', markersize=10, label = stations[i])


for x,y,label in zip (source_lons, source_lats,sources):
 ax.text(x+0.005, y, label, fontsize=10, ha='left', va='center')
for x,y,label in zip (station_lons, station_lats,stations):
 if label == "Glassport":
  ax.text(x+0.0035, y, label, fontsize=10, ha='left', va='bottom') 
 else: 
  ax.text(x+0.005, y, label, fontsize=10, ha='left', va='center') 

# Set the aspect ratio of the axis
ax.set_aspect('equal')

# Add a basemap using contextily
crs = 'EPSG:4326'
ctx.add_basemap(ax, crs=crs, source = basemap)
ax.set_axis_off()

# Add a legend of two different markers
legend_handles = [
    Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=10, label='Emission Source'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',markersize=10, label='Monitoring Station')
]

# Add legend with custom handles
ax.legend(handles=legend_handles)
plt.tight_layout()
# plt.savefig("map.png")

