
from define import NueralNet
from define import train_model

import torch
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as func

model = NeuralNet()  # Recreate the model instance

model.load_state_dict(torch.load('models/digit_recog.pth'))

model.eval()  # Set to eval mode (disables dropout, etc.)
