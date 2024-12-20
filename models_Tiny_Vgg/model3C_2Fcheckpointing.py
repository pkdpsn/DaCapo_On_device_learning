from torch import nn
from torch.utils.checkpoint import checkpoint


class MNIST_CNN(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()

        # First Convolutional Block (we'll save this output)
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Second Convolutional Block (we'll use checkpointing)
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Third Convolutional Block (we'll use checkpointing)
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Flatten Layer
        self.flatten = nn.Flatten()

        # Fully Connected Layers
        self.dense_1 = nn.Linear(in_features=hidden_units * 28 * 28, out_features=hidden_units)  # Adjust for input size
        self.dense_2 = nn.Linear(in_features=hidden_units, out_features=output_shape)

    def forwardsuboptimal(self, x):
        # Forward pass through the first convolutional block without checkpointing

        x = checkpoint(self.conv_block_1, x)
        # Use checkpointing for the second and third convolutional blocks
        x = checkpoint(self.conv_block_2, x)
        x = checkpoint(self.conv_block_3, x)

        # Flatten the output for the dense layers
        x = self.flatten(x)

        # Use checkpointing for the dense layers
        x = checkpoint(self.dense_1, x)
        x = checkpoint(self.dense_2, x)

        return x

    def forwardsub(self, x):
        # Forward pass through the first convolutional block without checkpointing
        x = self.conv_block_1(x)

        # Save the output of the first convolutional block
        self.saved_conv1_output = x.clone()

        # Use checkpointing for the second and third convolutional blocks
        x = checkpoint(self.conv_block_2, x)
        x = checkpoint(self.conv_block_3, x)

        # Flatten the output for the dense layers
        x = self.flatten(x)

        # Use checkpointing for the dense layers
        x = checkpoint(self.dense_1, x)
        x = checkpoint(self.dense_2, x)

        return x

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.flatten(x)
        x = checkpoint(self.dense_1, x, use_reentrant=False)
        x = checkpoint(self.dense_2, x, use_reentrant=False)
        return x