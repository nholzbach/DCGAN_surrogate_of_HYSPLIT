The files for this folder are too big to be uploaded to Github. Instead, you can find them on Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8366009.svg)](https://doi.org/10.5281/zenodo.8366009)


The following files are there:
1. full_input.pt
2. tensors.h5
3. full_input_100epoch.pt
4. tensors_100epochs.h5
5. model_state_threshold_11579.pth
6. state11579_full_input.pt

The 'full_input' is the input vectors used and in their particular order, the tensors are the generated images, in the same order as the full_input vectors. The state .pth file contains the state dictionary of the model at a certain iteration and can be used in `load_from_statedict.py` to generate data.
