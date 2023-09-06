#!/bin/bash

#SBATCH -t 48:00:00                                 # wall clock time
#SBATCH --partition=thin                            # partition type, normal=default                         
#SBATCH --mail-type=BEGIN,END                       # email settings
#SBATCH --mail-user=nina.holzbach@student.uva.nl 

python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-01"
python $HOME/hysplit_data/main_edit.py remove_files "2022-12-01"
wait
python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-02"
python $HOME/hysplit_data/main_edit.py remove_files "2022-12-02"
wait
python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-03"
python $HOME/hysplit_data/main_edit.py remove_files "2022-12-03"
wait
python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-04"
python $HOME/hysplit_data/main_edit.py remove_files "2022-12-04"
wait
python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-05"
python $HOME/hysplit_data/main_edit.py remove_files "2022-12-05"
wait
python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-06"
python $HOME/hysplit_data/main_edit.py remove_files "2022-12-06"
wait
python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-07"
python $HOME/hysplit_data/main_edit.py remove_files "2022-12-07"
wait
python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-08"
python $HOME/hysplit_data/main_edit.py remove_files "2022-12-08"
wait
python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-09"
python $HOME/hysplit_data/main_edit.py remove_files "2022-12-09"
wait
python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-10"
python $HOME/hysplit_data/main_edit.py remove_files "2022-12-10"
wait
python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-11"
python $HOME/hysplit_data/main_edit.py remove_files "2022-12-11"
wait
python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-12"
python $HOME/hysplit_data/main_edit.py remove_files "2022-12-12"
wait
python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-13"
python $HOME/hysplit_data/main_edit.py remove_files "2022-12-13"
wait
python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-14"
python $HOME/hysplit_data/main_edit.py remove_files "2022-12-14"
wait
python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-15"
python $HOME/hysplit_data/main_edit.py remove_files "2022-12-15"
wait
python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-16"
python $HOME/hysplit_data/main_edit.py remove_files "2022-12-16"
wait
python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-17"
python $HOME/hysplit_data/main_edit.py remove_files "2022-12-17"
wait
python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-18"
python $HOME/hysplit_data/main_edit.py remove_files "2022-12-18"
wait
python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-19"
python $HOME/hysplit_data/main_edit.py remove_files "2022-12-19"
wait
python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-20"
python $HOME/hysplit_data/main_edit.py remove_files "2022-12-20"
wait
python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-21"
python $HOME/hysplit_data/main_edit.py remove_files "2022-12-21"
wait
python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-22"
python $HOME/hysplit_data/main_edit.py remove_files "2022-12-22"
wait
python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-23"
python $HOME/hysplit_data/main_edit.py remove_files "2022-12-23"
wait
python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-24"
python $HOME/hysplit_data/main_edit.py remove_files "2022-12-24"
wait
python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-25"
python $HOME/hysplit_data/main_edit.py remove_files "2022-12-25"
wait
python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-26"
python $HOME/hysplit_data/main_edit.py remove_files "2022-12-26"
wait
python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-27"
python $HOME/hysplit_data/main_edit.py remove_files "2022-12-27"
wait
python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-28"
python $HOME/hysplit_data/main_edit.py remove_files "2022-12-28"
wait
python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-29"
python $HOME/hysplit_data/main_edit.py remove_files "2022-12-29"
wait
python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-30"
python $HOME/hysplit_data/main_edit.py remove_files "2022-12-30"
wait
python $HOME/hysplit_data/main_edit.py run_hysplit  "2022-12-31"
python $HOME/hysplit_data/main_edit.py remove_files "2022-12-31"
wait
zip $HOME/hysplit_data/results/2022-12.zip $HOME/hysplit_data/results/run_2022-12-*
wait
rm $HOME/hysplit_data/results/run_2022-12*
