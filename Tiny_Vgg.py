import torch
# import os
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from utils import *
from model import MNIST_CNN
# from model_eff import MNIST_CNN
device = 'cpu'## "cuda" if torch.cuda.is_available() else "cpu"

verbose = True
image_plot = True
if verbose:
  print(device)
print("Libraries imported - ready to use PyTorch", torch.__version__)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
BATCH_SIZE = 128
EPOCHS = 10


# Load the data

train_data = datasets.MNIST(root="data", train=True, download=True,transform=ToTensor(),target_transform=None)
test_data = datasets.MNIST(root="data",train=False,download=True,transform=ToTensor(),target_transform=None)

if verbose:
  print("Train data size: ", len(train_data))
  print("Test data size: ", len(test_data))
  print("Data loaded")

class_names = train_data.classes
if verbose:
  print("Classes: ", class_names)

# Display the images
# image = train_data[0][0]
# plt.imshow(image.squeeze(), cmap="gray")
# plt.axis(False)

train_dataloader = DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)
test_dataloader = DataLoader(dataset=test_data,batch_size=BATCH_SIZE,shuffle=False)

## define model and optimizer
model = MNIST_CNN(input_shape=1,hidden_units=10,output_shape=len(class_names)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),lr=0.1)
for EPOCH in range(EPOCHS):
    print(f"Epoch {EPOCH}\n --------------------------")
    train_step(model=model,
            accuracy_fn=accuracy_fn,
            loss_fn=loss_fn,
            data_loader=train_dataloader,
            optimizer=optimizer)
    test_step(model=model,
              accuracy_fn=accuracy_fn,
              loss_fn=loss_fn,
              data_loader=test_dataloader,
              optimizer=optimizer)