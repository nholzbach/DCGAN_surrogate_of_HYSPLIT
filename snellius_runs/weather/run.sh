#!/bin/bash

#SBATCH -t 04:30:00                                 # wall clock time
#SBATCH --partition=thin                            # partition type, normal=default                         
#SBATCH --mail-type=BEGIN,END                       # email settings
#SBATCH --mail-user=nina.holzbach@student.uva.nl 


python -u weather_extract3full.py 2016 12 1 31 > log_2016-12.out
#python -u weather_extract3.py 2021 2 1 28 > log_2021-02.out
#python -u weather_extract3.py 2021 3 1 31 > log_2021-03.out
#python -u weather_extract3.py 2021 4 1 30 > log_2021-04.out
#python -u weather_extract3.py 2021 5 1 31 > log_2021-05.out
#python -u weather_extract3.py 2020 6 1 30 > log_2020-06.out
#python -u weather_extract3.py 2022 2 1 28 > log_2022-02.out
#python -u weather_extract3.py 2018 8 1 31 > log_2018-08.out
#python -u weather_extract3.py 2018 9 1 30 > log_2018-09.out
#python -u weather_extract3.py 2018 10 1 31 > log_2018-10.out
#python -u weather_extract3.py 2021 11 1 30 > log_2021-11.out
#python -u weather_extract3.py 2021 12 1 31 > log_2021-12.out
