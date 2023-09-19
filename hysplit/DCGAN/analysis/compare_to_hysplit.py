# Code for comparing the output of a dcgan run to the original HYSPLIT output, at an hourly resolution 
# This code produces 4 images:
# 1. A plot of the DCGAN output, HYSPLIT output for a given hour. It will print the correlation coefficient, cross correlation, average rmse and SSIM score
# 2. A plot of the DCGAN output, HYSPLIT output, AND citizen reports for a given hour
# 3. A scatter plot of the correlation coefficient and SSIM score for all hours in the training data
# 4. A plot of the distribution of correlation coefficients and SSIM scores for all hours in the training data
# See the variables to adjust section to adjust the test number (of the DCGAN), iteration number, and hour of interest. Plot the citizen reports by setting CR to True

from tqdm import tqdm
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
import torch
import h5py
from skimage.metrics import structural_similarity as ssim
from scipy.signal import correlate2d

sns.set_style("whitegrid")
sns.set_context("talk", font_scale=2.0)

# VARIABLES TO ADJUST
test_num = 22
iter_num = 11579
dcganPATH = f'analysis/grids/run{test_num}/'
yr_interest, mo_interest, day_interest, hr_interest = 2022, 10, 17, 0
# True if you want to make another plot with citizen reports too
CR = False

# the file names may differ slightly so uncomment the correct one
# input_info = torch.load(f'../results_images/test_{test_num}/state{iter_num}_full_input.pt')
# input_info = torch.load(f'../results_images/test_{test_num}/full_input_100epoch.pt')
input_info = torch.load(f'../results_images/test_{test_num}/full_input.pt')

dcgan_output = []
# with h5py.File(f'../results_images/test_{test_num}/tensors_state{iter_num}.h5', 'r') as hf:
# with h5py.File(f'../results_images/test_{test_num}/tensors_100epochs.h5', 'r') as hf:
with h5py.File(f'../results_images/test_{test_num}/tensors.h5', 'r') as hf:
    for dataset_name in hf:
        dcgan_output.append(hf[dataset_name][:])
        
        
# scale the dcgan output
for i in range(len(dcgan_output)):
    grid = dcgan_output[i].squeeze()
    flat_grid = grid.reshape(-1)
    #inverse min max scaling
    flat_grid_rescaled = (flat_grid - (-1.0)) * (35.0 - 0.0) / (1.0 - (-1.0)) + 0.0
    dcgan_output[i] = flat_grid_rescaled.reshape(64,64)
  

# get dates
dates = []
for i in range(len(input_info)):
    dates.append(input_info[i][-9])
# inverse scale
dates = np.array(dates)
dates = dates * (1680300000 - 1470002400) + 1470002400
# convert to datetime
dates = pd.to_datetime(dates, unit='s')

# This does the analysis for the one specified hour of interest
for dcgan, time in zip(dcgan_output, dates):

 yr, mo,day, hr = time.year, time.month, time.day, time.hour
 if yr==yr_interest and mo==mo_interest and day==day_interest and hr ==hr_interest:
  filename = f'../input_data/training/training_{yr}_{str(mo).zfill(2)}_{day}_{hr}.csv'
  # plot_dates.append(time)
  hysplit = pd.read_csv(filename, skiprows=1,header=None)
  hysplit = hysplit.to_numpy()
  corr = np.corrcoef(dcgan.ravel(), hysplit.ravel())
  correlation_coefficient = corr[0, 1] 
  print("corr: " , correlation_coefficient)
  
 #   cross correlation
  cross_corr = correlate2d(dcgan, hysplit, mode='same', boundary='fill')
  print("cross_corr: ", cross_corr)

  # plot the two grids
  fig , ax = plt.subplots(1,2, figsize = (20,10), sharex=True, sharey=True)
  ax[0].imshow(dcgan, cmap = 'rocket_r')
  ax[0].set_title('DCGAN')
  sm1 = plt.cm.ScalarMappable(cmap = 'rocket_r')
  sm1.set_array(dcgan)
  cbar = plt.colorbar(sm1, ax=ax[0], shrink=0.4)
  
  ax[1].imshow(hysplit, cmap = 'rocket_r')
  ax[1].set_title('HYSPLIT')
  sm2 = plt.cm.ScalarMappable(cmap = 'rocket_r')
  sm2.set_array(hysplit)
  cbar = plt.colorbar(sm2, ax=ax[1], shrink=0.4)
  plt.tight_layout()
  plt.savefig(f'images/run{test_num}-{iter_num}_hysplit_comparison_{yr,mo,day,hr}.pdf')

  # compare the two grids
  rmse = np.sqrt((dcgan - hysplit)**2)
  average_rmse = np.mean(rmse)
  print(average_rmse)
  
  ssim_score = ssim(dcgan, hysplit, data_range=dcgan.max() - dcgan.min())
  print('ssim_score', ssim_score)

  
 if CR==True:
  fig , ax = plt.subplots(1,3, figsize = (30,10), sharex=True, sharey=True)
  ax[0].imshow(dcgan, cmap = 'rocket_r')
  ax[0].set_title('DCGAN')
  sm1 = plt.cm.ScalarMappable(cmap = 'rocket_r')
  sm1.set_array(dcgan)
  cbar = plt.colorbar(sm1, ax=ax[0], shrink=0.4)
  
  ax[1].imshow(hysplit, cmap = 'rocket_r')
  ax[1].set_title('HYSPLIT')
  sm2 = plt.cm.ScalarMappable(cmap = 'rocket_r')
  sm2.set_array(hysplit)
  cbar = plt.colorbar(sm2, ax=ax[1], shrink=0.4)

  cr_gpd = gpd.read_file(f'../../../citizen_data/grids/average/2022-10-22 22:00:00.geojson')
  cr = cr_gpd['smell value'].values.reshape(64,64).T
  ax[2].imshow(cr, cmap = 'rocket_r')
  ax[2].set_title('Citizen Reports')
  sm3 = plt.cm.ScalarMappable(cmap = 'rocket_r')
  sm3.set_array(cr)
  cbar = plt.colorbar(sm3, ax=ax[2], shrink=0.4)
  
  plt.tight_layout()
  plt.savefig(f'images/run{test_num}-{iter_num}_3grid_comparison_{yr,mo,day,hr}.pdf')




# Running for all days in the training data and just getting results, not plotting the grids 
correlation_coeff = []
plot_dates = []
ssim_scores = []
cross_corrs = []
for dcgan, time in tqdm(zip(dcgan_output, dates), total=len(dcgan_output), desc="Processing DCGAN outputs"):

    yr, mo,day, hr = time.year, time.month, time.day, time.hour
    filename = f'../input_data/training/training_{yr}_{str(mo).zfill(2)}_{day}_{hr}.csv'
    plot_dates.append(time)
    hysplit = pd.read_csv(filename, skiprows=1,header=None)
    hysplit = hysplit.to_numpy()
    corr = np.corrcoef(dcgan.ravel(), hysplit.ravel())
    correlation_coefficient = corr[0, 1] 
    correlation_coeff.append(correlation_coefficient)

    # cross correlation
    cross_corr = correlate2d(dcgan, hysplit, mode='same', boundary='fill')
    cross_corrs.append(cross_corr)
    # SSIM
    ssim_score = ssim(dcgan, hysplit, data_range=dcgan.max() - dcgan.min())
    ssim_scores.append(ssim_score)
   
# plot the correlation and ssim scores
fig, ax = plt.subplots(1,1, figsize=(50,10))
plt.scatter(plot_dates, correlation_coeff, alpha=0.5)
plt.scatter(plot_dates, ssim_scores, color='r', alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Hour')
plt.ylabel('Correlation Coefficient/SSIM Score')
plt.title(f'Image Similarity analysis for DCGAN vs HYSPLIT for {test_num}')
plt.savefig(f'images/run{test_num}_full_measures.pdf')

# plot the distribution of the correlation coefficients
fig, ax = plt.subplots(1,1, figsize=(10,10))
sns.histplot(correlation_coeff, ax=ax, kde=True, color='b', label='Correlation Coefficient', binwidth=0.03)
sns.histplot(ssim_scores, ax=ax, kde=True, color='r', label='SSIM Score', binwidth=0.03)
plt.xlim(- 1, 1)
plt.ylim(0, 2000)
plt.xticks()
plt.yticks()
plt.xlabel('Correlation Coefficient/SSIM Score')
plt.ylabel('Count')
# plt.savefig(f'images/run{test_num}_hysplit_comparison_iter{iter_num}.pdf')
# plt.savefig(f'images/run{test_num}_hysplit_comparison_100epochs.pdf')
plt.savefig(f'images/run{test_num}_hysplit_comparison_distribution.pdf')


