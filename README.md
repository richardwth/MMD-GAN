# MMD-GAN with Repulsive Loss Function
GAN: generative adversarial nets; MMD: maximum mean discrepancy; TF: TensorFlow

This repository contains codes for MMD-GAN and the repulsive loss proposed in the following paper:

[1] Improving MMD-GAN Training with Repulsive Loss Function.  Under review as a conference paper at ICLR 2019. URL: https://openreview.net/forum?id=HygjqjR9Km.

## About the code
The code was written along with my learning of Python and GAN and contains many other models I have tried, so I apologize if you find it messy and confusing. The core idea is to define the neural network architecture as dictionaries to quickly test different models.

For your interest,
1. DeepLearning/my_sngan/SNGan defines how the model is trained and evaluated. 
2. GeneralTools/graph_func contains metrics for evaluating generative models (Line 1595). 
3. GeneralTools/math_func contains spectral normalization (Line 397) and a variety of loss functions for GAN (Line 2088). The repulsive loss can be found at Line 2505; the repulsive loss with bounded kernel (referred to as rmb) at Line 2530.
4. my_test_* contain the model architecture, hyperparameters, and training procedures. 

### How to use
1. Modify GeneralTools/misc_func accordingly; 
2. Read Data/ReadMe.md; download and prepare the datasets;
3. Run my_test_* with proper hyperparameters.

## About the algorithms
Here we summarize the algorithms and tricks in case you want to implement the algorithms yourself. 

The paper [1] proposed three tricks:
1. Repulsive loss

![equation](https://latex.codecogs.com/gif.latex?\inline&space;L_G=\sum_{i\ne&space;j}k_D(x_i,x_j)-2\sum_{i\ne&space;j}k_D(x_i,y_j)&plus;\sum_{i\ne&space;j}k_D(y_i,y_j))

![equation](https://latex.codecogs.com/gif.latex?\inline&space;L_D=\sum_{i\ne&space;j}k_D(x_i,x_j)-\sum_{i\ne&space;j}k_D(y_i,y_j))

where ![equation](https://latex.codecogs.com/gif.latex?\inline&space;x_i,x_j) - real samples, ![equation](https://latex.codecogs.com/gif.latex?\inline&space;y_i,y_j) - generated samples, ![equation](https://latex.codecogs.com/gif.latex?\inline&space;k_D) - kernel formed by the discriminator ![equation](https://latex.codecogs.com/gif.latex?\inline&space;D) and kernel ![equation](https://latex.codecogs.com/gif.latex?\inline&space;k). 

2. Bounded kernel (used only in ![equation](https://latex.codecogs.com/gif.latex?L_D))

![equation](https://latex.codecogs.com/gif.latex?\inline&space;k_D^{b}(x_i,x_j)&space;=\exp(-\frac{1}{2\sigma^2}\min(\left&space;\|&space;D(x_i)-D(x_j)&space;\right&space;\|^2,&space;b_u)))

![equation](https://latex.codecogs.com/gif.latex?\inline&space;k_D^{b}(y_i,y_j)&space;=\exp(-\frac{1}{2\sigma^2}\max(\left&space;\|&space;D(y_i)-D(y_j)&space;\right&space;\|^2,&space;b_l)))

3. Power iteration for convolution (used in spectral normalization)

At each iteration, for convolution kernel ![equation](https://latex.codecogs.com/gif.latex?\inline&space;W_c), do ![equation](https://latex.codecogs.com/gif.latex?\inline&space;u=\text{conv}(W_c,v)), ![equation](https://latex.codecogs.com/gif.latex?\inline&space;v=\text{transpose-conv}(W_c,u)), and ![equation](https://latex.codecogs.com/gif.latex?\inline&space;\hat{v}=v/\left&space;\|&space;v&space;\right&space;\|).


In spectral normalization of our paper [1], we directly estimate the spectral norm of a convolution kernel, which empirically is larger than the spectral norm of the reshaped kernel estimated in [2]. Thus, our spectral normalization imposes a stronger penalty than [2]'s. As a result, in our case, the norm of the signal will tend to decrease in each layer because:
- It is unlikely for the signal to coincide with the first eigenvector ("eigentensor") of the convolutional kernel
- It is likely the activation function (leaky ReLU or ReLU) reduces the norm of the signal. 

Consequently, the discriminator outputs tend to be the same for any inputs (where the outputs are just the biases propagating through the network). Therefore, we found it essential to multiply the signal with a constant **C** > 1 after each spectral normalization (GeneralTools/layer_func Line 822). The GAN performance (using the repulsive loss, MMD loss and hinge loss) seems to be sensitive to **C** as:
- small **C** may limit the magnitude of model weights and gradients and slow down the training.
- large **C** reduces the penalty of the spectral norm and may result in unstable training.

I did not mention this in the paper [1] where we used **C**=1.82 empirically. Later I found **C**=![equation](http://latex.codecogs.com/gif.latex?64^{1/L}) seems to provide more stable results across different learning rate combinations, where L is the number of discriminator layers. I will provide more details and discussion on this as soon as I get a chance to revise the paper. 

## Reference
[1] Improving MMD-GAN Training with Repulsive Loss Function.  Under review as a conference paper at ICLR 2019. URL: https://openreview.net/forum?id=HygjqjR9Km. \
[2] Takeru Miyato, Toshiki Kataoka, Masanori Koyama, and Yuichi Yoshida. Spectral normalization
for generative adversarial networks. In ICLR, 2018
