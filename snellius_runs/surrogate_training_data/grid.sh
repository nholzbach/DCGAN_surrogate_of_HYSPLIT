#!/bin/bash

#SBATCH -t 10:00:00                                 # wall clock time
#SBATCH --partition=thin                            # partition type, normal=default                         
#SBATCH --mail-type=BEGIN,END                       # email settings
#SBATCH --mail-user=nina.holzbach@student.uva.nl 

python -u gridding.py 2022_11 > log_2022-11.out 
#python -u gridding.py 2018_09 > log_2018-09.out
#python -u gridding.py 2018_08 > log_2018-08.out
#python -u gridding.py 2018_10 > log_2018-10.out
#python -u gridding.py 2021_05 > log_2021-05.out
#python -u gridding.py 2020_06 > log_2020-06.out
#python -u gridding.py 2019_01 > log_2019-01.out
#python -u gridding.py 2019_02 > log_2019-02.out
#python -u gridding.py 2022_03 > log_2022-03.out
#python -u gridding.py 2019_04 > log_2019-04.out
#python -u gridding.py 2019_05 > log_2019-05.out
#python -u gridding.py 2019_06 > log_2019-06.out
#python -u gridding.py 2019_07 > log_2019-07.out
#python -u gridding.py 2019_08 > log_2019-08.out
#python -u gridding.py 2022_09 > log_2022-09.out
#python -u gridding.py 2019_11 > log_2019-11.out
#python -u gridding.py 2019_12 > log_2019-12.out
#python -u gridding.py 2022_10 > log_2022-10.out
