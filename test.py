
from define import NueralNet
from define import test_model

import torch
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as func
from torchvision import datasets, transforms


EPOCHS = 5
#uses class from "define.py"
model = NueralNet()  # Recreate the model instance

criterion = nn.CrossEntropyLoss()

# load previously saved model
model.load_state_dict(torch.load('models/digit_recog.pth'))

model.eval()  # Set to eval mode (disables dropout, etc.)

test_model(model,criterion, EPOCHS)

