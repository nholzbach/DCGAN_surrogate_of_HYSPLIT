#!/bin/bash

#SBATCH -t 15:30:00                                 # wall clock time
#SBATCH --partition=thin                            # partition type, normal=default                         
#SBATCH --mail-type=BEGIN,END                       # email settings
#SBATCH --mail-user=nina.holzbach@student.uva.nl 

python -u grid_comparison.py 2022 6 > CR_log_2022-6.out
python -u grid_comparison.py 2022 12 > CR_log_2022-12.out
python -u grid_comparison.py 2018 9 > CR_log_2018-9.out
python -u grid_comparison.py 2018 8 > CR_log_2018-8.out
python -u grid_comparison.py 2021 5 > CR_log_2021-5.out
python -u grid_comparison.py 2019 1 > CR_log_2019-1.out
python -u grid_comparison.py 2019 2 > CR_log_2019-2.out
python -u grid_comparison.py 2019 3 > CR_log_2019-3.out
python -u grid_comparison.py 2019 4 > CR_log_2019-4.out
python -u grid_comparison.py 2019 5 > CR_log_2019-5.out
python -u grid_comparison.py 2019 6 > CR_log_2019-6.out
python -u grid_comparison.py 2019 7 > CR_log_2019-7.out
python -u grid_comparison.py 2019 8 > CR_log_2019-8.out
python -u grid_comparison.py 2019 9 > CR_log_2019-9.out
python -u grid_comparison.py 2019 10 > CR_log_2019-10.out
python -u grid_comparison.py 2019 11 > CR_log_2019-11.out
python -u grid_comparison.py 2019 12 > CR_log_2019-12.out

