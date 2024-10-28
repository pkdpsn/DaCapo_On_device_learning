from torch import nn

# Device configuration
device = 'cpu'  # 'cuda' if torch.cuda.is_available() else 'cpu'

class MNIST_CNN(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()

        # First Convolutional Block
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )

        # Second Convolutional Block
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3,stride=1,padding=1), nn.ReLU()
        )

        # Third Convolutional Block
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )

        # Fourth Convolutional Block
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,stride=1,padding=1), nn.ReLU()
        )

        # Fifth Convolutional Block
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )

        # Sixth Convolutional Block
        self.conv_block_6 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3, stride=1, padding=1), nn.ReLU()
        )

        # Seventh Convolutional Block
        self.conv_block_7 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )

        # Eighth Convolutional Block
        self.conv_block_8 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )

        # Flatten Layer
        self.flatten = nn.Flatten()

        # Fully Connected Layers
        self.dense_1 = nn.Linear(in_features=hidden_units * 28 * 28, out_features=hidden_units)  # Adjust for input size
        self.dense_2 = nn.Linear(in_features=hidden_units, out_features=output_shape)

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.conv_block_5(x)
        x = self.conv_block_6(x)
        x = self.conv_block_7(x)
        x = self.conv_block_8(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        return x
