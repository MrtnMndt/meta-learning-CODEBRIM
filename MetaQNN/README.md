# CODEBRIM_MetaQNN
This is our open-source implementation of the Q-learning based MetaQNN neural architecture search algorithm to find suitable convolutional neural network architectures for our challenging multi-class multi-target *CODEBRIM* bridge-defect recognition dataset (it will work with other datasets by simply adding corresponding data loaders). The entire code has been written in *Python 3.5.2* (although the code should work in principle with other *Python 3* versions as well) and *PyTorch 1.0.0*. 

## Installing code dependencies
We have added the main code dependencies in a requirements text file. For full reproducibility in the future we have further added another requirements_full_build text file with the exact configuration of our machine (note that you will not need a majority of these packages for this repository). 

`pip3 install -r requirements.txt`

## Running a search
A search can be conducted by simply executing

`python3 main.py -t 1 --dataset-path PATH_TO_CODEBRIM_DATASET`

This will launch the MetaQNN search with 200 architectures and the exact hyperparameters as specified in the paper. Before running the search, the *CODEBRIM* dataset needs to be downloaded from https://zenodo.org/record/2620293 , unzipped and the path to this directory should be substituted for *PATH_TO_CODEBRIM_DATASET*. All necessary search and training hyperparameters are exposed in the command line parser that can be found in *lib/cmdparser.py*.

## Resuming an incomplete search
An incomplete search can be resumed by executing

`python3 main.py -t 1 --dataset-path PATH_TO_CODEBRIM_DATASET --q-values-csv-path PATH_TO_LAST_Q_VALUES --replay-buffer-csv-path PATH_TO_LAST_REPLAY_BUFFER`

where the additional arguments, *PATH_TO_LAST_Q_VALUES* and *PATH_TO_LAST_REPLAY_BUFFER* are paths to the most recently saved *Q-values* and *replay buffer* csv files during the incomplete search.

## Search space definition
The minimum and maximum amount of layers can be specified through the command line parser. Other parameters, such as the different convolutional filter sizes, amount of units in fully-connected layers or filters in the convolution, spatial pyramidal pooling (SPP) sizes and the epsilon schedule (number of architectures to search) are defined in a config file that can be found at `lib/Models/state_space_parameters.py`.

## GPU requirements and sequentializing of mini-batches
We have made sure to be able to execute the architecture search and make everything trainable on a single GPU. In practice we have implemented functions to estimate the required amount of memory of a neural architecture for the given mini-batch size while it is getting built (before starting to train and running out of memory). If the available GPU memory is surpassed then the mini-batch is sequentialized, i.e. divided into smaller mini-batches that get executed in sequence and gradients get accumulated until the overall mini-batch size is reached and an update is conducted. The amount of overall SGD updates thus always stays the same, but the time it takes to train grows with less available GPU memory. While it should be possible to execute the code on any GPU, we therefore highly recommend using a GPU with at least 8 Gb of memory.  

We note that the amount of required memory is unfortunately non-trivial to estimate in an exact fashion as CUDNN and other factors can induce variations from machine to machine. We have thus used a heuristic that slightly overestimates the required memory to be on the safe side. If you are reading this and have the knowledge to implement this in an exact fashion in PyTorch, please do not hesitate to contact us or open a pull-request.

## Retraining an architecture from the search
To retrain and revaluate an architecture from a previous search, one may execute

 `python3 main.py -t 2 --dataset-path PATH_TO_CODEBRIM_DATASET --replay-buffer-csv-path PATH_TO_LAST_REPLAY_BUFFER --fixed-net-index-no INDEX_NO_OF_ARC`
 
Here, the additional parameter, *INDEX_NO_OF_ARC* denotes the index of the architecture configuration which needs to be retrained, as per the indexing in the replay buffer.
