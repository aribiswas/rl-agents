"""
Deep Q network model (PyTorch).
@author: abiswas
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.torch_utils import to_tensor


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
        x = to_tensor(state)

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


class QValueFunction(nn.Module):

    def __init__(self, state_dim, action_dim, is_image_obs=False):
        """
        Initialize a Q-Value function.

        Parameters
        ----------
        state_dim : list, tuple
            Dimension of state.
        action_dim : list, tuple
            Dimension of action.

        Returns
        -------
        None.

        Example:
        q = QValueFunction(state_dim=(4,), action_dim=(2,))
        q = QValueFunction(state_dim=(20,20,3), action_dim=(4,))
        """

        self.is_image_obs = is_image_obs
        self.state_dim = state_dim
        self.action_dim = action_dim
        super(QValueFunction, self).__init__()

        # state path
        if self.is_image_obs:
            # Conv2d transformations:
            # Wout = (Win + 2*P - D*(K-1) -1)/S + 1
            # Wout: output size
            # Win: input size
            # P: padding
            # D: dilation
            # K: kernel size
            # S: stride
            self.conv1 = nn.Conv2d(state_dim[2], 16, kernel_size=(4, 4), stride=(2, 2))
            self.bn1 = nn.BatchNorm2d(16)
            convw = (state_dim[0] + 2 * 0 - 1 * (4 - 1) - 1) / 2 + 1
            convh = (state_dim[1] + 2 * 0 - 1 * (4 - 1) - 1) / 2 + 1
            self.sfc1 = nn.Linear(int(convw * convh * 16), 64)
        else:
            self.sfc1 = nn.Linear(state_dim[0], 64)
        self.sfc2 = nn.Linear(64, 64)

        # action path
        self.afc1 = nn.Linear(action_dim[0], 64)
        self.afc2 = nn.Linear(64, 64)

        # common path
        self.cfc1 = nn.Linear(64 * 2, 64)
        self.cfc2 = nn.Linear(64, 1)

    def forward(self, state, action):
        """
        Forward pass through the network.

        :param state: state (tensor)
        :param action: action (tensor)
        :return: Q value (tensor)
        """

        xs = to_tensor(state)
        xa = to_tensor(action)

        # state path
        if self.is_image_obs:
            # Inputs here are NxHxWxC (batch) or HxWxC (single)
            #
            # For conv2d, inputs must have dimension NxCxHxW
            # N = number of batches
            # C = number of channels
            # H = height of image
            # W = width of image
            N = xs.shape[-4] if len(xs.shape) == 4 else 1
            H = xs.shape[-3]
            W = xs.shape[-2]
            C = xs.shape[-1]
            xs = xs.reshape((N, C, H, W))

            xs = F.relu(self.bn1(self.conv1(xs)))
            xs = xs.view(-1, self.layer_size[0])  # flatten the tensor
        else:
            xs = F.tanh(self.sfc1(xs))
        xs = F.tanh(self.sfc2(xs))

        # action path
        xa = F.tanh(self.afc1(xa))
        xa = F.tanh(self.afc2(xa))

        # common path
        xc = torch.cat((xs, xa), dim=1)
        xc = F.relu(self.cfc1(xc))
        xc = self.cfc2(xc)

        return xc


class DeterministicActor(nn.Module):

    def __init__(self, state_dim, action_dim, is_image_obs=False):

        self.is_image_obs = is_image_obs
        self.state_dim = state_dim
        self.action_dim = action_dim
        super(DeterministicActor, self).__init__()

        # state path
        if self.is_image_obs:
            # Conv2d transformations:
            # Wout = (Win + 2*P - D*(K-1) -1)/S + 1
            # Wout: output size
            # Win: input size
            # P: padding
            # D: dilation
            # K: kernel size
            # S: stride
            self.conv1 = nn.Conv2d(state_dim[2], 16, kernel_size=(4, 4), stride=(2, 2))
            self.bn1 = nn.BatchNorm2d(16)
            convw = (state_dim[0] + 2 * 0 - 1 * (4 - 1) - 1) / 2 + 1
            convh = (state_dim[1] + 2 * 0 - 1 * (4 - 1) - 1) / 2 + 1
            self.fc1 = nn.Linear(int(convw * convh * 16), 64)
        else:
            self.fc1 = nn.Linear(state_dim[0], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim[0])

    def forward(self, state):
        """
        Perform forward pass through the network.
        """
        x = to_tensor(state)

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
        x = F.tanh(self.fc3(x))
        return x
