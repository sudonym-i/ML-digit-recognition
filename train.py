from define import NueralNet
from define import train_model

import torch
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as func

EPOCHS = 5

model = NueralNet()

# Use cross-entropy for classification
criterion = nn.CrossEntropyLoss()

# Use Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_model(model, criterion, optimizer, EPOCHS)

torch.save(model.state_dict(), 'models/digit_recog.pth')

