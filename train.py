from define import NueralNet
from define_train import train_model

import torch
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as func

EPOCHS = 5

#class from "define.py"
model = NueralNet()

# Using cross-entropy for classification
criterion = nn.CrossEntropyLoss()

# Using Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_model(model, criterion, optimizer, EPOCHS)

torch.save(model.state_dict(), 'models/digit_recog.pth')

