
import torch
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter


#for logging data to tensorboard
writer = SummaryWriter("training_runs/mnist_digit_recog")


def test_model(model, criterion, num_epochs):

    writer = SummaryWriter("test_runs/mnist_digit_recog")

    transform = transforms.ToTensor()

    test_dataset = datasets.MNIST('./test_data', train=False, download=True, transform=transform)
    
    test_loader = DataLoader(test_dataset, batch_size=60, shuffle=True)

    for epoch in range(num_epochs):  # Train for x epochs

        total = 0
        correct = 0

        for batch_idx, (images, labels) in enumerate(test_loader):
        
            # Forward pass
            outputs = model(images)

            loss = criterion(outputs, labels)

            # this is just for modelling preformance
            if(torch.max(outputs, 1) == labels):
                correct+=1
            total+=1

        # log for tensorboard
        writer.add_scalar("Loss/train", loss.item() / len(test_loader), epoch)
        writer.add_scalar("Accuracy/train", correct / total, epoch)

        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
