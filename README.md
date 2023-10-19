# Progressive-Growing-GAN-Generative-Adversarial-Network-Implementatio

## Table of Contents

- [Overview](#overview)
- [Code Structure](#code-structure)
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
