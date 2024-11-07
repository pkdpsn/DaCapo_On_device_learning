from torch import nn
from torch.utils.checkpoint import checkpoint

# Device configuration
device = 'cpu'  # 'cuda' if torch.cuda.is_available() else 'cpu'


class MNIST_CNN(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()

        # First Convolutional Block (we'll save this output)
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Second to Fifteenth Convolutional Blocks (we'll use checkpointing for these)
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.conv_block_6 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.conv_block_7 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.conv_block_8 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.conv_block_9 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.conv_block_10 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.conv_block_11 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.conv_block_12 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.conv_block_13 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.conv_block_14 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.conv_block_15 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Flatten Layer
        self.flatten = nn.Flatten()

        # Fully Connected Layers
        self.dense_1 = nn.Linear(in_features=hidden_units * 28 * 28, out_features=hidden_units)  # Adjust for input size
        self.dense_2 = nn.Linear(in_features=hidden_units, out_features=output_shape)

    def forwardsuboptimal(self, x):
        # Use checkpointing for all convolutional layers
        x = checkpoint(self.conv_block_1, x)  # Layer 1
        x = checkpoint(self.conv_block_2, x)  # Layer 2
        x = checkpoint(self.conv_block_3, x)  # Layer 3
        x = checkpoint(self.conv_block_4, x)  # Layer 4
        x = checkpoint(self.conv_block_5, x)  # Layer 5
        x = checkpoint(self.conv_block_6, x)  # Layer 6
        x = checkpoint(self.conv_block_7, x)  # Layer 7
        x = checkpoint(self.conv_block_8, x)  # Layer 8
        x = checkpoint(self.conv_block_9, x)  # Layer 9
        x = checkpoint(self.conv_block_10, x)  # Layer 10
        x = checkpoint(self.conv_block_11, x)  # Layer 11
        x = checkpoint(self.conv_block_12, x)  # Layer 12
        x = checkpoint(self.conv_block_13, x)  # Layer 13
        x = checkpoint(self.conv_block_14, x)  # Layer 14
        x = checkpoint(self.conv_block_15, x)  # Layer 15
        # Flatten the output for the dense layers
        x = self.flatten(x)
        # Use checkpointing for the dense layers
        x = checkpoint(self.dense_1, x)  # Dense Layer 1
        x = checkpoint(self.dense_2, x)  # Dense Layer 2
        return x

    def forward(self, x):
        # Forward pass through the first convolutional block without checkpointing
        x = self.conv_block_1(x)

        # Save the output of the first convolutional block
        self.saved_conv1_output = x.clone()

        # Use checkpointing for the remaining convolutional blocks
        x = checkpoint(self.conv_block_2, x)
        x = checkpoint(self.conv_block_3, x)
        x = checkpoint(self.conv_block_4, x)
        x = checkpoint(self.conv_block_5, x)
        x = checkpoint(self.conv_block_6, x)
        x = checkpoint(self.conv_block_7, x)
        x = checkpoint(self.conv_block_8, x)
        x = checkpoint(self.conv_block_9, x)
        x = checkpoint(self.conv_block_10, x)
        x = checkpoint(self.conv_block_11, x)
        x = checkpoint(self.conv_block_12, x)
        x = checkpoint(self.conv_block_13, x)
        x = checkpoint(self.conv_block_14, x)
        x = checkpoint(self.conv_block_15, x)

        # Flatten the output for the dense layers
        x = self.flatten(x)

        # Use checkpointing for the dense layers
        x = checkpoint(self.dense_1, x)
        x = checkpoint(self.dense_2, x)

        return x

    def forwardsub(self,x):
        x = self.conv_block_1(x)  # Layer 1
        x = checkpoint(self.conv_block_2, x)  # Layer 2
        x = checkpoint(self.conv_block_3, x)  # Layer 3
        x = self.conv_block_4(x) # Layer 4
        x = checkpoint(self.conv_block_5, x)  # Layer 5
        x = checkpoint(self.conv_block_6, x)  # Layer 6
        x = self.conv_block_7(x)
        x = checkpoint(self.conv_block_8, x)  # Layer 8
        x = checkpoint(self.conv_block_9, x)  # Layer 9
        x = self.conv_block_10(x)
        x = checkpoint(self.conv_block_11, x)  # Layer 11
        x = checkpoint(self.conv_block_12, x)  # Layer 12
        x = self.conv_block_13(x)
        x = checkpoint(self.conv_block_14, x)  # Layer 14
        x = checkpoint(self.conv_block_15, x)  # Layer 15
        x = self.flatten(x)
        x = checkpoint(self.dense_1, x)  # Dense Layer 1
        x = checkpoint(self.dense_2, x)  # Dense Layer 2
        return x
