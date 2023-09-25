# Citizen data of reported odours, from [Smell Pittsburgh](https://smellpgh.org/) project

- `edda_citizenreports.py` performs exploratory data analysis of the citizen dataset and produces a number of figures.
 - The raw data can be found in the *raw* folder.
 - The resulting figures are found in the *results* folder.
- `gridding_CR.py` grids the smell reports and saves the grids as geojson files. 3 time resolutions can be done: hourly, morning/afternoon, and daily. These grids can be saved to geojson files. Check line 137 in the script for more information.
 - Examples of these geojsons are found in the *grid* folder.
- `test_results.txt` contains some elementary results from an analysis of the text citizen data.
 
  
