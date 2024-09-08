from torch import nn
import torch

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
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_block_2 = nn.Sequential(
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

    def forward(self, x):
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
        self.saved_dense1_output = x.clone()

        # Forward pass through the second dense layer
        x = self.dense_2(x)

        return x

    # def backward(self, grad_output):
    #     # Recompute activations for the first dense layer
    #     self.saved_dense1_output.requires_grad = False
    #     self.saved_dense1_output = self.flatten(self.saved_conv1_output)
    #     self.saved_dense1_output = self.dense_1(self.saved_dense1_output)
    #
    #     # Recompute activations for the first convolutional block
    #     self.saved_conv1_output.requires_grad = False
    #     self.saved_conv1_output = torch.randn_like(self.saved_conv1_output)  # Placeholder for actual recomputation
    #
    #     # Manually perform the backward pass
    #     grad_input = torch.autograd.grad(self.saved_dense1_output, self.parameters(), grad_outputs=grad_output)
    #     return grad_input


# Test checkpointing and forward pass
# if __name__ == "__main__":
#     model = MNIST_CNN(input_shape=1, hidden_units=10, output_shape=10).to(device)
#     print("Model architecture:")
#     print(model)
#
#     # Dummy input for testing
#     dummy_input = torch.randn(1, 1, 28, 28).to(device)
#     output = model(dummy_input)
#
#     print("Output of the model:", output)
#     print("Checkpointed conv1 output:", model.saved_conv1_output)
#     print("Checkpointed dense1 output:", model.saved_dense1_output)
