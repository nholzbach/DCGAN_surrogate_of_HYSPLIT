#!/bin/bash

#SBATCH -t 15:30:00                                 # wall clock time
#SBATCH --partition=thin                            # partition type, normal=default                         
#SBATCH --mail-type=BEGIN,END                       # email settings
#SBATCH --mail-user=nina.holzbach@student.uva.nl 

python -u morning-afternoon.py 2022 6 > morningafternoon_log_2022-6.out
python -u morning-afternoon.py 2022 12 > morningafternoon_log_2022-12.out
python -u morning-afternoon.py 2018 9 > morningafternoon_log_2018-9.out
python -u morning-afternoon.py 2018 8 > morningafternoon_log_2018-8.out
python -u morning-afternoon.py 2021 5 > morningafternoon_log_2021-5.out
python -u morning-afternoon.py 2019 1 > morningafternoon_log_2019-1.out
python -u morning-afternoon.py 2019 2 > morningafternoon_log_2019-2.out
python -u morning-afternoon.py 2019 3 > morningafternoon_log_2019-3.out
python -u morning-afternoon.py 2019 4 > morningafternoon_log_2019-4.out
python -u morning-afternoon.py 2019 5 > morningafternoon_log_2019-5.out
python -u morning-afternoon.py 2019 6 > morningafternoon_log_2019-6.out
python -u morning-afternoon.py 2019 7 > morningafternoon_log_2019-7.out
python -u morning-afternoon.py 2019 8 > morningafternoon_log_2019-8.out
python -u morning-afternoon.py 2019 9 > morningafternoon_log_2019-9.out
python -u morning-afternoon.py 2019 10 > morningafternoon_log_2019-10.out
python -u morning-afternoon.py 2019 11 > morningafternoon_log_2019-11.out
python -u morning-afternoon.py 2019 12 > morningafternoon_log_2019-12.out

