# MMD-GAN with Repulsive Loss Function
GAN: generative adversarial nets; MMD: maximum mean discrepancy; TF: TensorFlow

This repository contains codes for MMD-GAN and the repulsive loss proposed in the following paper:

Improving MMD-GAN Training with Repulsive Loss Function.  Under review as a conference paper at ICLR 2019. URL: https://openreview.net/forum?id=HygjqjR9Km.

## Repository structure
The codes here were written along with my learning of Python and GAN, so I apologise if you find them quite messy, out-of-date or informal. 

The respository mainly contains:
1. DeepLearning/my_sngan/SNGan \
This class defines how the model is trained and evaluated. 

2. GeneralTools
- graph_func has functions/classes for TF Optimizer, Session, Summary, checkpoint file as well as metrics for evaluating generative models.
- input_func has functions/classes for saving and reading data to/from tfrecords, and sampling from simple distributions.
- layer_func has functions/classes to define operations in each layer and network routines.
- math_func has functions/classes for basic math operations (like pairwise distance), spectral normalization, and a variety of loss functions for GAN.
- misc_func contains the basic info for machine, libraries and working directory

3. my_test_* \
These scripts contain the model architecture, hyperparameters, and training procedures. 

## How to use
1. Modify misc_func accordingly; download and prepare the datasets.
2. Run my_test_* with proper hyperparameters
