# Progressive-Growing-GAN-Generative-Adversarial-Network-Implementatio

## Table of Contents

- [Overview](#overview)
- [Code Structure](#code-structure)
- [Model](#Model)
- [Training Strategy](#training-strategy)
- [Key Achievements](#key-achievements)
- [How to Use](#how-to-use)
- [Dependencies](#dependencies)
- [License](#license)

## Overview

This repository contains an implementation of a Progressive Growing GAN, a generative model capable of producing high-resolution images from random noise vectors. The code is organized into several Python files, each serving a specific role in building and training the GAN. This README provides an overview of the code structure and how to use it.

## Code Structure

The code is organized into the following files:

1. `dataset.py`: Defines dataset paths and provides data loading functionality, including resizing, data augmentation, and normalization. It also includes a function to visualize sample images from the dataset.

2. `modules.py`: Contains the definition of key neural network modules used in the GAN model, including the generator and discriminator. These modules are essential for the architecture and training of the GAN.

3. `train.py`: Sets up hyperparameters, implements the training loop for both the generator and discriminator, and follows a progressive growing strategy. It aims to minimize the WGAN loss and enforce gradient penalties for stable training.

4. `predict.py`: Loads the trained generator model and generates sample images from random noise vectors. The generated images are saved and displayed for evaluation.

   ## Model
A **styleGAN** model is constructed and used for the given task, which is a state-of-the-art variation of the Generative Adversarial Network(GAN). Building on regular GAN models, styleGAN is designed to further improve the quality as well as the control of generated images by introducing several extra components:
- **Style Mapping**:
  Instead of directly feeding the latent space vector *z* into the generator, which might cause the issue of feature entanglement, styleGAN firstly converts the *z* to an intermediate latent space *w* (also known as the style factor) via a mapping network, in order to untangle the data distribution so that training the generator could be easier. 
  The style mapping technique is achieved by the *MappingNetwork* class in the *modules.py* file, which is essentially a network of 8 fully connected layers that takes in *z* and outputs *w*;
- **Adaptive Instance Normalization**:
  AdaIN is essentially a normalization technique that aligns the mean and variance of the content feature with that of the style feature in the generator of styleGAN model. It helps to modulate and manipulate generated images based on the style factor *w*.
  The Adaptive Instance Normalization technique is achieved by the *AdIN* class in the *modules.py* file;
- **Stochastic Variation as Noise Input**:
  Stochastic variation is introduced to different layers of the generator using scale/resolution-specific Gaussian noises, where the scale-specificity is achieved by the learnable scaling factor (represented as the *weight* variable in the code), allowing for fine details such as hairs and freckles to be generated.
  The stochastic variation technique is achieved by the *injectNoise* class in the *modules.py* file;
- **Weight-scaled Convolution**:
  In the styleGAN model, the weights of convolutional layers are all normalized by scaling the input to the convolution based on input channels and the kernel size, so that the training procedure is more stable.
  The weight-scaled convolution technique is achieved by the *WSConv2d* class in the *modules.py* file;

*Note: the style mapping, AdaIN and stochastic variation techniques are used to construct the blocks of generator only, while the weight-scaled convolution is applied to both generator and discriminater.*

- **Progressive Training**:
  The training of styleGAN starts with low-resolution images (4 \* 4 in this case) and progressively increases the resolution by adding new layers until it reaches the resolution of the original images to be resembled (256 \* 256). This approach not only accelerates but also stabilizes the training process.
  The progressive training technique is supported by both generator and discriminater and achieved in the *train.py* file.


## Training Strategy

The Progressive Growing GAN employs a training strategy where the resolution of generated images is gradually increased during training. This progressive approach allows the model to produce high-resolution images by initially generating low-resolution images and incrementally adding detail as training progresses.

## Key Achievements

The implementation successfully trains a Progressive Growing GAN to generate high-quality images from random noise vectors. This demonstrates the effectiveness of the GAN architecture and its ability to produce realistic and diverse images.

## How to Use

To use this code, follow these steps:

1. Clone this repository to your local machine.

2. Install the necessary dependencies (listed below).

3. Run the training script `train.py` to train the Progressive Growing GAN.

4. After training, you can use the `predict.py` script to generate sample images with the trained model.

## Dependencies

Ensure you have the following dependencies installed:

- Python 3
- PyTorch
- torchvision
- Matplotlib

You can install the required Python packages using `pip`:

```bash
pip install torch torchvision matplotlib
