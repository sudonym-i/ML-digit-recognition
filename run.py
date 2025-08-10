
from define import NueralNet
from define_train import train_model

import torch
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as func

#uses class from "define.py"
model = NueralNet()  # Recreate the model instance

# load previously saved model
model.load_state_dict(torch.load('models/digit_recog.pth'))

model.eval()  # Set to eval mode (disables dropout, etc.)
