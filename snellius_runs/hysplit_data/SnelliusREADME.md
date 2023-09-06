In order to run the HYSPLIT simualtions for Pittsburgh on a supercomputer, the following files can be used.
These files have been copied from the CREATE SmellPittsburgh project and modified.

To check before running:
- Change all paths in `main.py` and `cached_hysplit_run_lib.py`
- Change the path in which hysplit software is found in `automate_plume_viz.py`
- Check all the settings defined in `main.py` and adjust as required.

HYSPLIT configuration settings can be changed in `cached_hysplit_run_lib.py`, from line 492 onwards.

The simulation can be run with a bash script, similar to `one_month_job.sh`.
Since the output files take up a considerable amount of space, they are zipped into one folder with all dates from a month combined. 
