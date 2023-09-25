# DCGAN as a surrogate model for data generation of HYSPLIT simulations

- `DCGAN.py` contains all the code needed to train the DCGAN and generate data immediately. It is recommended to run this file by segmenting it into cells and having a more interactive jupyter-like environment, however it can be run all at once simply through a terminal. 
There are a number of variables that need to be adjusted so please read the comments in the code carefully. 

- `load_from_statedict.py` loads a saved state dictionary and then data is generated from that. 

- *analysis* folder contains evaluation and validation methods. See folder for more details.

- *input_data* folder contains some of the training data used that was generated with code in the *snellius_runs* folder. This training data consists of an informed vector and corresponding HYSPLIT image for each timestamp.

- *results_images* folder contians some examples of results from training the DCGAN. Test 22 is provided, with involves 16347 data entries and was training for 100 epochs, with a batchsize of 128. Please read the READme in there for more information.

- *stats* folder contains pickled statistics that were saved from training tests. Metrics such as generator and discriminator losses, time of training, number of iterations etc.
