#!/bin/bash

#SBATCH -t 48:00:00                                 # wall clock time
#SBATCH --partition=thin                            # partition type, normal=default                         
#SBATCH --mail-type=BEGIN,END                       # email settings
#SBATCH --mail-user=nina.holzbach@student.uva.nl 

#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-01"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-12-01"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-02"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-12-02"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-03"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-12-03"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-04"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-12-04"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-05"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-12-05"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-06"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-12-06"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-07"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-12-07"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-08"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-12-08"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-09"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-12-09"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-10"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-12-10"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-11"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-12-11"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-12"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-12-12"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-13"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-12-13"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-14"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-12-14"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-15"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-12-15"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-16"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-12-16"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-17"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-12-17"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-18"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-12-18"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-19"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-12-19"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-20"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-12-20"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-21"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-12-21"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-22"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-12-22"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-23"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-12-23"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-24"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-12-24"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-25"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-12-25"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-26"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-12-26"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-27"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-12-27"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-28"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-12-28"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-29"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-12-29"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-30"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-12-30"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-31"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-12-31"
#qwait
#q
#qzip $HOME/hysplit_data/results/2022-12.zip $HOME/hysplit_data/results/run_2022-12-*
#qwait
#qrm $HOME/hysplit_data/results/run_2022-12*
#qwait
#python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-09-01"
#python $HOME/hysplit_data/main_edit.py remove_files "2022-09-01"
#wait
#python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-09-02"
#python $HOME/hysplit_data/main_edit.py remove_files "2022-09-02"
#wait
#python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-09-03"
#python $HOME/hysplit_data/main_edit.py remove_files "2022-09-03"
#wait
#python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-09-04"
#python $HOME/hysplit_data/main_edit.py remove_files "2022-09-04"
#wait
#python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-09-05"
#python $HOME/hysplit_data/main_edit.py remove_files "2022-09-05"
#wait
#python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-09-06"
#python $HOME/hysplit_data/main_edit.py remove_files "2022-09-06"
#wait
#python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-09-07"
#python $HOME/hysplit_data/main_edit.py remove_files "2022-09-07"
#wait
#python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-09-08"
#python $HOME/hysplit_data/main_edit.py remove_files "2022-09-08"
#wait
#python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-09-09"
#python $HOME/hysplit_data/main_edit.py remove_files "2022-09-09"
#wait
#python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-09-10"
#python $HOME/hysplit_data/main_edit.py remove_files "2022-09-10"
#wait
#python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-09-11"
#python $HOME/hysplit_data/main_edit.py remove_files "2022-09-11"
#wait
#python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-09-12"
#python $HOME/hysplit_data/main_edit.py remove_files "2022-09-12"
#wait
#python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-09-13"
#python $HOME/hysplit_data/main_edit.py remove_files "2022-09-13"
#wait
#python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-09-14"
#python $HOME/hysplit_data/main_edit.py remove_files "2022-09-14"
#wait
#python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-09-15"
#python $HOME/hysplit_data/main_edit.py remove_files "2022-09-15"
#wait
#python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-09-16"
#python $HOME/hysplit_data/main_edit.py remove_files "2022-09-16"
#wait
#python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-09-17"
#python $HOME/hysplit_data/main_edit.py remove_files "2022-09-17"
#wait
#python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-02-31"
#python $HOME/hysplit_data/main_edit.py remove_files "2022-02-31"
#wait
#python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-02-30"
#python $HOME/hysplit_data/main_edit.py remove_files "2022-02-30"
#wait
#python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-02-29"
#python $HOME/hysplit_data/main_edit.py remove_files "2022-02-29"
#wait
#python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-03-12"
#python $HOME/hysplit_data/main_edit.py remove_files "2022-03-12"
#wait
#python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-09-22"
#python $HOME/hysplit_data/main_edit.py remove_files "2022-09-22"
#wait
#python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-09-23"
#python $HOME/hysplit_data/main_edit.py remove_files "2022-09-23"
#wait
#python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-09-24"
#python $HOME/hysplit_data/main_edit.py remove_files "2022-09-24"
#wait
#python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-09-25"
#python $HOME/hysplit_data/main_edit.py remove_files "2022-09-25"
#wait
#python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-09-26"
#python $HOME/hysplit_data/main_edit.py remove_files "2022-09-26"
#wait
#python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-09-27"
#python $HOME/hysplit_data/main_edit.py remove_files "2022-09-27"
#wait
#python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-09-28"
#python $HOME/hysplit_data/main_edit.py remove_files "2022-09-28"
#wait
#python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-09-29"
#python $HOME/hysplit_data/main_edit.py remove_files "2022-09-29"
#wait
#python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-09-30"
#python $HOME/hysplit_data/main_edit.py remove_files "2022-09-30"
#wait
##python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-09-31"
##python $HOME/hysplit_data/main_edit.py remove_files "2022-09-31"
##wait
#zip $HOME/hysplit_data/results/2022-09.zip $HOME/hysplit_data/results/run_2022-09-*
#wait
#rm $HOME/hysplit_data/results/run_2022-09*
#wait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-11-01"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-11-01"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-11-02"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-11-02"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-11-03"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-11-03"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-11-04"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-11-04"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-11-05"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-11-05"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-11-06"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-11-06"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-11-07"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-11-07"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-11-08"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-11-08"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-11-09"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-11-09"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-11-10"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-11-10"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-11-11"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-11-11"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-11-12"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-11-12"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-11-13"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-11-13"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-11-14"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-11-14"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-11-15"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-11-15"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-11-16"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-11-16"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-11-17"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-11-17"
#qwait
#qpython $HOME/hysplit_data/main_edit.py run_hysplit  "2022-11-18"
#qpython $HOME/hysplit_data/main_edit.py remove_files "2022-11-18"
#qwait
python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-11-19"
python $HOME/hysplit_data/main_edit.py remove_files "2022-11-19"
wait
python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-11-20"
python $HOME/hysplit_data/main_edit.py remove_files "2022-11-20"
wait
python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-11-21"
python $HOME/hysplit_data/main_edit.py remove_files "2022-11-21"
wait
python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-11-22"
python $HOME/hysplit_data/main_edit.py remove_files "2022-11-22"
wait
python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-11-23"
python $HOME/hysplit_data/main_edit.py remove_files "2022-11-23"
wait
python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-11-24"
python $HOME/hysplit_data/main_edit.py remove_files "2022-11-24"
wait
python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-11-25"
python $HOME/hysplit_data/main_edit.py remove_files "2022-11-25"
wait
python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-11-30"
python $HOME/hysplit_data/main_edit.py remove_files "2022-11-30"
wait
python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-11-26"
python $HOME/hysplit_data/main_edit.py remove_files "2022-11-26"
wait
python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-11-27"
python $HOME/hysplit_data/main_edit.py remove_files "2022-11-27"
wait
python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-11-28"
python $HOME/hysplit_data/main_edit.py remove_files "2022-11-28"
wait
python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-11-29"
python $HOME/hysplit_data/main_edit.py remove_files "2022-11-29"
wait
zip $HOME/hysplit_data/results/2022-11.zip $HOME/hysplit_data/results/run_2022-11-*
wait
rm $HOME/hysplit_data/results/run_2022-11*


echo !!!DONE!!!
