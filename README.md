# MMD-GAN with Repulsive Loss Function
GAN: generative adversarial nets; MMD: maximum mean discrepancy; TF: TensorFlow

This repository contains codes for MMD-GAN and the repulsive loss proposed in the following paper:

[1] Improving MMD-GAN Training with Repulsive Loss Function.  Under review as a conference paper at ICLR 2019. URL: https://openreview.net/forum?id=HygjqjR9Km.

## Regarding the codes
The codes here were written along with my learning of Python and GAN, so I apologise if you find them messy and confusing. The core idea is to define the neural network architecture as dictionaries to quickly test differnt models.

For your interest,
1. DeepLearning/my_sngan/SNGan defines how the model is trained and evaluated. 
2. GeneralTools/graph_func contains metrics for evaluating generative models. GeneralTools/math_func contains spectral normalization and a variety of loss functions for GAN.
3. my_test_* contain the model architecture, hyperparameters, and training procedures. 

### How to use
1. Modify misc_func accordingly; download and prepare the datasets.
2. Run my_test_* with proper hyperparameters

## Regarding the algorithms
In spectral normalization of paper [1], we directly estimate the spectral norm of a convolution kernel, which empirically is larger than the spectral norm of the reshaped kernel estimated in [2]. Thus, our spectral normalization imposes a stronger pernalization than the original paper [2]. As a result, the norm of the signal will tend to decrease in each layer because:
- It is unlikely for the signal to coincide with the first eigenvector of the kernel
- It is likely the activation function (leaky ReLU or ReLU) decrease the norm of signal. 

We found it essential to multiply a constant larger than 1 after each 


## Reference
[1] Improving MMD-GAN Training with Repulsive Loss Function.  Under review as a conference paper at ICLR 2019. URL: https://openreview.net/forum?id=HygjqjR9Km.
[2] Takeru Miyato, Toshiki Kataoka, Masanori Koyama, and Yuichi Yoshida. Spectral normalization
for generative adversarial networks. In ICLR, 2018
