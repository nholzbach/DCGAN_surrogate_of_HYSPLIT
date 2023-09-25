# Analysis of DCGAN surrogate model
This was two part:
- Image comparison with HYSPLIT results
- Monitoring station and citizen report (CR) validation performance

Files:
- `CR_plotting.py` - plots are made from the CR analysis, from the data saved in *stats*, and the Jenson-Shannon divergence is calculated to compare the performance of the HYSPLIT validation and the DCGAN validation.
- `compare_to_hysplit.py` - code for comparing the output of a dcgan run to the original HYSPLIT output, at an hourly resolution. Two image simiarity measures are used: correlation coefficient and SSIM.
- `cr_analysis.py` - this is the code that actually does the analysis of the citizen reports and the DCGAN output. Results from here are saved in the *stats* folder.
- `point_analysis.py` - this code performs the monitoring station analysis for the DCGAN results. Jenson-Shannon divergence and KS tests are performed to compare the DCGAN's performance to the HYSPLIT's performance.

Folders:
- images contains all figures produced
- stats contains some example csv files of the morning/afternoon analysis
