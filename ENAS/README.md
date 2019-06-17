# CODEBRIM_ENAS: Efficient Neural Architecture Search via Parameter Sharing

This implementation is a fork from the original authors' implementation of "Efficient Neural Architecture Search via Parameter Sharing", ICML (2018) in TensorFlow that can be found at https://github.com/melodyguan/enas 

### License and modifications
Because this implementation is originally forked from the project mentioned above, we have reproduced their license in the license file. This license is in addition to our license specified for the overall CODEBRIM project if you are using it in the context of CODEBRIM. 

The main modifications we have made to the original code are some changes in search space and hyper-parameters, adaptation to the multi-target classification scenario and corresponding changes in optimization, addition of the CODEBRIM dataloader, visualization of the DAG and some minor fixes. The majority of the code thus remains the same as in the original repository. If you are using this version of the code, please also consider citing the original work. 

### Requirements
The requirements are the exact same as in the original ENAS repository, we have not changed them.

Unfortunately this means that the main requirement is both an older Python and TensorFlow version

* Python 2.7
* TensorFlow 1.4.0

## Run the CODEBRIM experiments

To run the CODEBRIM experiments download the CODEBRIM dataset at: [https://zenodo.org/record/2620293](https://zenodo.org/record/2620293). Depending on where you download it to you will have to specify the path in `src/AEROBI/data_utils.py` and replace the path in the path argument of the data loader class (we apologize for this being hardcoded).

To run the ENAS search experiments please use the following scripts:
```
./scripts/cifar10_macro_search.sh
```

An architecture for a neural network with `N` layers consists of `N` parts, indexed by `1, 2, 3, ..., N`. Part `i` consists of:

* A number in `[0, 1, 2, 3, 4, 5]` that specifies the operation at layer `i`-th, corresponding to `conv_3x3`, `separable_conv_3x3`, `conv_5x5`, `separable_conv_5x5`, `average_pooling`, `max_pooling`.
* A sequence of `i - 1` numbers, each is either `0` or `1`, indicating whether a skip connection should be formed from a the corresponding past layer to the current layer.

You can find an example in `./scripts/AEROBI_macro_final.sh`. Because the weights are shared during the search, once the search is over you should specify the model you want to retrain according to the string in the `macro_final.sh` file to re-train the final model and obtain the final performance. 

If you find the architecture strings to be too difficult to interpret, we have added visualization scripts below. 

## Visualization
We have added a jupyter notebook for visualization of the DAG search progress. It can be found in the `scripts` directory. Before you use the notebook to generate a video, you will have to use the `extractArcString.py` to extract all the architecture strings from your search output file (that also contains all the accuracies etc.). You will get a sequence of images and a video analogous to the one shown below as depicted in the appendix of our paper.

![](imgs/ENAS-1.pdf)
