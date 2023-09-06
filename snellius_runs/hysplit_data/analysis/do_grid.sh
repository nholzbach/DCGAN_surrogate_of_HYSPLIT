#!/bin/bash

#SBATCH -t 10:30:00                                 # wall clock time
#SBATCH --partition=thin                            # partition type, normal=default                         
#SBATCH --mail-type=BEGIN,END                       # email settings
#SBATCH --mail-user=nina.holzbach@student.uva.nl 

python -u daily_grid.py 2022-11 > grid_log_2022-11.out
#python -u daily_grid.py 2022-9 > grid_log_2022-9.out
#python -u daily_grid.py 2022-4 > grid_log_2022-4.out
#python -u daily_grid.py 2022-7 > grid_log_2022-7.out
#python -u daily_grid.py 2018-10 > grid_log_2018-10.out
#python -u daily_grid.py 2020-6 > grid_log_2020-6.out
#python -u daily_grid.py 2020-7 > grid_log_2020-7.out
#python -u daily_grid.py 2021-3 > grid_log_2021-3.out
#python -u daily_grid.py 2021-4 > grid_log_2021-4.out
#python -u daily_grid.py 2022-1 > grid_log_2022-1.out
#python -u daily_grid.py 2022-2 > grid_log_2022-2.out
#python -u daily_grid.py 2022-3 > grid_log_2022-3.out
#python -u daily_grid.py 2022-5 > grid_log_2022-5.out
