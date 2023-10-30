import torch
import torch.nn as nn

def build_nn(in_dim:int, hidden_dims:tuple, out_dim:int):
    """
        Create a fully connected neural network.
        :param hidden_dims  : hidden layer dimensions.
        :param out_dim      : output dimension.
    """
    hidden_dims.append(in_dim)
    layers = []
    for i in range(len(hidden_dims)-2):
        layers.append(nn.Linear(in_features=hidden_dims[i-1], out_features=hidden_dims[i]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(in_features=hidden_dims[-2], out_features=out_dim))
    return nn.Sequential(*layers)


def build_1d_cnn(in_dim:int, out_dim:int, kernel_size=3):
    """
        Create a 1D convolutional neural network.
        Used as an time series processor for time-series data.
        : param hidden_channels : hidden dimensions of the CNN.
        : param out_dim         : output dimension of the CNN.
        : param kernel_size     : kernel/filter width and height.
    """
    layers = []
    layers.append(nn.Conv1d(in_channels=in_dim, out_channels=out_dim, \
                            kernel_size=kernel_size, padding="same"))
    return nn.Sequential(*layers)


def build_2d_cnn(hidden_channels:list, out_dim:int, kernel_size=3):
    """
        Create a 2D convolutional neural network.
        Used as an image preprocessor for image data (MNIST derivatives, SPRITES)
            : param hidden_channels : hidden dimensions of the CNN.
            : param out_dim         : output dimension of the CNN.
            : param kernel_size     : kernel/filter width and height.
    """
    hidden_channels.append(out_dim)
    layers = []
    for i in range(len(hidden_channels)):
        layers.append(nn.Conv2d(in_channels=hidden_channels[i-1], out_channels=hidden_channels[i], \
                                kernel_size=kernel_size, padding="same"))
        layers.append(nn.BatchNorm2d(hidden_channels[i]))
        layers.append(nn.ReLU())

    return nn.Sequential(*layers)