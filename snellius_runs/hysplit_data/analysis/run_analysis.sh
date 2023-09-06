#!/bin/bash

#SBATCH -t 10:30:00                                 # wall clock time
#SBATCH --partition=thin                            # partition type, normal=default                         
#SBATCH --mail-type=BEGIN,END                       # email settings
#SBATCH --mail-user=nina.holzbach@student.uva.nl 


python -u point_analysis.py 2019-01 > new_log_2019-01.out
python -u point_analysis.py 2019-04 > new_log_2019-04.out
python -u point_analysis.py 2019-05 > new_log_2019-05.out
python -u point_analysis.py 2019-06 > new_log_2019-06.out
python -u point_analysis.py 2019-02 > new_log_2019-02.out
python -u point_analysis.py 2019-03 > new_log_2019-03.out
python -u point_analysis.py 2019-11 > new_log_2019-11.out
python -u point_analysis.py 2019-12 > new_log_2019-12.out
python -u point_analysis.py 2019-07 > new_log_2019-07.out
python -u point_analysis.py 2019-08 > new_log_2019-08.out
python -u point_analysis.py 2019-09 > new_log_2019-09.out
python -u point_analysis.py 2019-10 > new_log_2019-10.out

