from torch import nn

device = 'cpu'  # Use "cuda" if torch.cuda.is_available() else "cpu"

class MNIST_CNN(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()

        # 5 Convolutional Blocks, each with one Conv2d layer followed by ReLU
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

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

        # Flatten Layer
        self.flatten = nn.Flatten()

        # Calculate the input size for dense_1
        # Assuming the input size is 28x28 (like MNIST), we need to compute the output size
        # After 5 convolutional layers (without pooling), the size remains the same if padding is used
        self.dense_1 = nn.Linear(in_features=hidden_units * 28 * 28, out_features=hidden_units)  # Adjusted for input size
        self.dense_2 = nn.Linear(in_features=hidden_units, out_features=output_shape)

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.conv_block_5(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        return x
