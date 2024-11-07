from torch import nn
import torch
from torch.utils.checkpoint import checkpoint

device = 'cpu'  # "cuda" if torch.cuda.is_available() else "cpu"


class MNIST_CNN(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.flatten = nn.Flatten()
        self.dense_1 = nn.Linear(in_features=hidden_units * 7 * 7, out_features=hidden_units)
        self.dense_2 = nn.Linear(in_features=hidden_units, out_features=output_shape)

    def forward2(self, x):
        # Forward pass through the first convolutional block
        x = self.conv_block_1(x)
        # Save the output of the first convolutional block
        self.saved_conv1_output = x.clone()

        # Forward pass through the second convolutional block
        x = self.conv_block_2(x)

        # Flatten the output
        x = self.flatten(x)

        # Forward pass through the first dense layer
        x = self.dense_1(x)
        # Save the output of the first dense layer
        # self.saved_dense1_output = x.clone()

        # Forward pass through the second dense layer
        x = self.dense_2(x)

        return x

    def forward(self, x):
        # Forward pass through the first convolutional block without checkpointing
        x = self.conv_block_1(x)

        # Save the output of the first convolutional block
        self.saved_conv1_output = x.clone()

        # Forward pass through the second convolutional block with checkpointing
        x = checkpoint(self.conv_block_2, x)

        # Flatten the output for the dense layers
        x = self.flatten(x)

        # Use checkpointing for the dense layers to recompute fully during backward
        x = checkpoint(self.dense_1, x)
        x = checkpoint(self.dense_2, x)

        return x

