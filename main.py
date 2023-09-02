import sys
import subprocess
# implement pip as a subprocess:
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os 

# Used for saving plots
import matplotlib.image as mpimg

# Imports the training and testing modules
from train_cnn import training_cnn
from test_cnn import testing_cnn

# Creates directories to store results
def directory_creator():
  if not os.path.exists("./image_set"):
      os.makedirs("./image_set")

  if not os.path.exists("./results"):
    os.makedirs("./results")

  if not os.path.exists("./results/training_plots"):
    os.makedirs("./results/training_plots")

directory_creator()

# All of the parameters can be adjusted to get better values where needed
transform = transforms.Compose([
   transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
   transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
   transforms.RandomHorizontalFlip(p=0.5),
   transforms.ToTensor(),
   transforms.Normalize(mean = 0.5, std = 0.5)
])

"""
Prepares the train and test set by loading CIFAR10 dataset from torchvision.datasets
https://pytorch.org/vision/0.9/datasets.html#cifar
"""

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
classes = trainset.classes


# Use torch.utils.data.random_split() to make a validation set from the training set with 80:20 split.
train_size = int(0.8 * len(trainset))
valid_size = len(trainset) - train_size
trainset, validset = data.random_split(trainset, [train_size, valid_size])


# Sets a batch size
batch_size = 64

# Writes dataloader for train set
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

# Writes dataloader for test set
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

# Writes dataloader for validation set
validloader = torch.utils.data.DataLoader(validset, batch_size=64, shuffle=False, num_workers=2)


# Loads a random batch of images from the training set using the trainloader.
dataiter = iter(trainloader)
images, labels = next(dataiter)

# Show the images
grid = torchvision.utils.make_grid(images)
grid = grid / 2 + 0.5
grid_numpy = grid.numpy()
plt.imshow(np.transpose(grid_numpy, (1, 2, 0)))
plt.show()

# Saves the image in case the plot doesn't pop up (for example in an unconfigured WSL)
mpimg.imsave("./image_set/training_image_set.png", np.transpose(grid_numpy, (1, 2, 0)))

# Prints the ground truth class labels for these images
for i in range(batch_size):
  print("Image {} : {}".format(i+1,classes[labels[i]]))

class Noah(nn.Module):
    def __init__(self):
        super(Noah,self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.full1 = nn.Linear(in_features=400, out_features=120)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.full2 = nn.Linear(in_features=120, out_features=80)
        self.relu2 = nn.ReLU()
        self.full3 = nn.Linear(in_features=80, out_features=10)


    def forward(self, x):
        # Define the dataflow through the layers
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.view(-1, 16*5*5)
        x = self.full1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.full2(x)
        x = self.relu2(x)
        x = self.full3(x)

        return x
    
model = Noah()
# Defines the number of epochs
num_epochs = 20
training_cnn(model,trainloader,validloader,num_epochs)

testing_cnn(model,trainloader,testloader,batch_size,classes, num_epochs)



