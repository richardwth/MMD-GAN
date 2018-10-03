# MMD-GAN with Repulsive Loss Function
GAN: generative adversarial nets; MMD: maximum mean discrepancy; TF: TensorFlow

This repository contains codes for MMD-GAN and the repulsive loss proposed in the following paper:

[1] Improving MMD-GAN Training with Repulsive Loss Function.  Under review as a conference paper at ICLR 2019. URL: https://openreview.net/forum?id=HygjqjR9Km.

## Code structure
The codes here were written along with my learning of Python and GAN, so I apologise if you find them messy and confusing. The core idea is to define the neural network architecture as dictionaries to quickly test differnt models.

For your interest,
1. DeepLearning/my_sngan/SNGan defines how the model is trained and evaluated. 
2. GeneralTools
- graph_func contains metrics for evaluating generative models.
- math_func contains spectral normalization and a variety of loss functions for GAN.
3. my_test_* contain the model architecture, hyperparameters, and training procedures. 

## How to use
1. Modify misc_func accordingly; download and prepare the datasets.
2. Run my_test_* with proper hyperparameters

## Note
In spectral normalization of paper [1], we directly estimate the spectral norm of a convolution kernel, which empirically is larger than the spectral norm of the reshaped kernel. Thus, our spectral normalization impose a strong pernalization. As a result, the norm of the signal will definitely decrease in each layer. 


We found it essential
