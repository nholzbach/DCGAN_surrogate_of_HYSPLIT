# HYSPLIT simulations results and DCGAN surrogate model

Please read the READme within the *DCGAN* folder for more information. 

Files:
-`emission_scheme.py` - This code is used to preprocess and explore the sensor data. Then, plots the emission scheme for each station. Finally, missing data is imputed and files are created that can be used for the informed input vector for the DCGAN.
- `get_data.py` - This code downloads monitoring station data from the online server. 

Folders:
- *HYSPLIT_results* folder contains scripts for plotting the results for both validation procedures: Monitoring Station (fixed point) and Citizen Report driven.
The actual analysis scripts are found in the *snellius_runs* folder as they were also done on the supercomputer due to memory limitations.
- *sensor_data* folder contains both the raw data and processed data for each monitoring station. The `explantion_map.py` script produces "map.png", an illustration the geographical context of the monitoring stations and emission sources.



