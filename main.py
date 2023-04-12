# Importing the libraries
import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST # Training dataset
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

## Defining the layer of the Generator

def generator_layer(input_dim,output_dim):
    '''

    This function creates a layer of the generator neural network.

    Input:
    input_dim: It is the dimension of the layer's input, a scaler
    output_dim: It is the dimension of the layer's output vector

    Output:
    returns the output of the layer after linear transformation followed by batch normalisation
    and ReLU activation function

    '''
    return nn.Sequential(
        nn.Linear(input_dim,output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace = True)
    )