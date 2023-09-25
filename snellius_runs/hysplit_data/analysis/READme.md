These are all the files that perform analysis and a short description of what they do. Please also check the comments in each script for variables/paths/processes that need adjusting. 

- `daily_grid.py` - this crops and discretises a grid of the HYSPLIT simulations so that they are in the correct image format to be used to train the DCGAN. 
            run with `do_grid.sh`
- `grid_comparison.py` - this does an hourly resolution comparison of HYSPLIT with the citizen reports. Both the pathway and impact analysis are calculated here and saved to a csv. 
            run with `cr_analysis.sh`
- `morning-afternoon.py`- this performs a partitioned day resolution, instead of an hourly. 
            run with `cr_morningafternoon.sh`
- `point_analysis.py` - this performs the monitoring station fixed point analysis for the HYSPLIT results
            run with `run_analysis.sh`       
