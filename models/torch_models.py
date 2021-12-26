"""
Deep Q network model (PyTorch).
@author: abiswas
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QNetwork(nn.Module):

    def __init__(self, input_size, num_outputs, is_image_obs=False):
        """
        Initialize a Deep Q-network.

        Parameters
        ----------
        input_size : list, tuple
            Dimension of inputs (observations).
        num_outputs : list
            Dimension of outputs (actions).

        Returns
        -------
        None.

        Example:
        q = QNetwork(input_size=(4,), num_outputs=2)
        q = QNetwork(input_size=(20,20,3), num_outputs=4)
        """

        self.is_image_obs = is_image_obs
        self.input_size = input_size
        self.num_outputs = num_outputs

        super(QNetwork, self).__init__()

        # initialize layers
        if self.is_image_obs:
            # Conv2d transformations:
            # Wout = (Win + 2*P - D*(K-1) -1)/S + 1
            # Wout: output size
            # Win: input size
            # P: padding
            # D: dilation
            # K: kernel size
            # S: stride
            self.conv1 = nn.Conv2d(input_size[2], 16, kernel_size=(4, 4), stride=(2, 2))
            self.bn1 = nn.BatchNorm2d(16)
            convw = (input_size[0] + 2 * 0 - 1 * (4 - 1) - 1) / 2 + 1
            convh = (input_size[1] + 2 * 0 - 1 * (4 - 1) - 1) / 2 + 1
            self.layer_size = [int(convw * convh * 16), 64, 32, num_outputs]
        else:
            self.layer_size = [input_size[0], 64, 64, num_outputs]

        # linear layers
        self.fc1 = nn.Linear(self.layer_size[0], self.layer_size[1])
        self.fc2 = nn.Linear(self.layer_size[1], self.layer_size[2])
        self.fc3 = nn.Linear(self.layer_size[2], self.layer_size[3])

    def forward(self, state):
        """
        Forward pass through the network.
        Parameters
        ----------
        state : tensor
            Model input.
        Returns
        -------
        x : tensor
            Model output.
        """

        # convert to torch
        if isinstance(state, np.ndarray):
            x = torch.from_numpy(state).float()
        elif isinstance(state, torch.Tensor):
            x = state
        else:
            raise TypeError("Input must be a numpy array or torch Tensor.")

        # make forward pass through the network
        if self.is_image_obs:
            # Inputs here are NxHxWxC (batch) or HxWxC (single)
            #
            # For conv2d, inputs must have dimension NxCxHxW
            # N = number of batches
            # C = number of channels
            # H = height of image
            # W = width of image
            N = x.shape[-4] if len(x.shape) == 4 else 1
            H = x.shape[-3]
            W = x.shape[-2]
            C = x.shape[-1]
            x = x.reshape((N, C, H, W))

            x = F.relu(self.bn1(self.conv1(x)))
            x = x.view(-1, self.layer_size[0])  # flatten the tensor

        # continue with the linear layers
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x
