# MMD-GAN with Repulsive Loss Function
GAN: generative adversarial nets; MMD: maximum mean discrepancy; TF: TensorFlow

This repository contains codes for MMD-GAN and the repulsive loss proposed in the following paper:

[1] Improving MMD-GAN Training with Repulsive Loss Function.  Under review as a conference paper at ICLR 2019. URL: https://openreview.net/forum?id=HygjqjR9Km.

## About the code
The code was written along with my learning of Python and GAN and contains many other models I have tried, so I apologize if you find it messy and confusing. The core idea is to define the neural network architecture as dictionaries to quickly test different models.

For your interest,
1. DeepLearning/my_sngan/SNGan defines how the model is trained and evaluated. 
2. GeneralTools/graph_func contains metrics for evaluating generative models (Line 1594). GeneralTools/math_func contains spectral normalization (Line 397) and a variety of loss functions for GAN (including the proposed repulsive loss at Line 2501 and 2516).
3. my_test_* contain the model architecture, hyperparameters, and training procedures. 

### How to use
1. Modify GeneralTools/misc_func accordingly; download and prepare the datasets.
2. Run my_test_* with proper hyperparameters

## About the algorithms
In spectral normalization of our paper [1], we directly estimate the spectral norm of a convolution kernel, which empirically is larger than the spectral norm of the reshaped kernel estimated in [2]. Thus, our spectral normalization imposes a stronger penalty than [2]'s. As a result, in our case, the norm of the signal will tend to decrease in each layer because:
- It is unlikely for the signal to coincide with the first eigenvector ("eigentensor") of the convolutional kernel
- It is likely the activation function (leaky ReLU or ReLU) reduces the norm of the signal. 

Consequently, the discriminator outputs tend to be the same for any inputs (where the outputs are just the biases propagating through the network). Therefore, we found it essential to multiply the signal with a constant **C** > 1 after each spectral normalization. Actually, using a fixed kernel scale, the MMD loss seems to be sensitive to **C** as 
- small **C** may limit the magnitude of pair-wise distance thus the boundary of kernel values.
- large **C** reduces the penalty of the spectral norm and may result in unstable training.

I did not mention this in the paper [1] where we used 1.82 empirically. Later I found 1.5 seems to provide more stable results on CIFAR-10 dataset across different learning rate combinations. I will provide more details and discussion on this as soon as I get a chance to revise the paper. 

## Reference
[1] Improving MMD-GAN Training with Repulsive Loss Function.  Under review as a conference paper at ICLR 2019. URL: https://openreview.net/forum?id=HygjqjR9Km. \
[2] Takeru Miyato, Toshiki Kataoka, Masanori Koyama, and Yuichi Yoshida. Spectral normalization
for generative adversarial networks. In ICLR, 2018
