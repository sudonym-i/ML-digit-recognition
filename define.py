
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("training_runs/mnist_digit_recog")



class NueralNet(nn.Module):
    def __init__(self):
        super().__init__()
        # layer one
        self.layer_1 = nn.Linear(28*28, 128)
        #output layer
        self.layer_2 = nn.Linear(128, 10)
    
    def forward(self, x):
            #flatten down to one dimension, inferring batch size with -1
        x = x.view(-1, 28*28)

        x = func.silu(self.layer_1(x))
        x = self.layer_2(x)
        return x



def train_model(model, criterion, optimizer, num_epochs):
    
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST('./training_data', train=True, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=60, shuffle=True)

    for epoch in range(num_epochs):  # Train for x epochs

        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
        
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if(torch.max(outputs, 1) == labels):
                correct+=1
            total+=1

        writer.add_scalar("Loss/train", loss.item() / len(train_loader), epoch)
        writer.add_scalar("Accuracy/train", correct / total, epoch)

        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
