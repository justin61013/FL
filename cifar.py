from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import Tensor
from torchvision.datasets import CIFAR10
import flwr as fl
import product_dataset
# import efficientnet_pytorch
from config import *


# build efficientnet model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.model = EfficientNet.from_pretrained('efficientnet-b0')
        self.model._fc = nn.Linear(1280, 10)
    def forward(self, x):
        x = self.model(x)
        return x

DATA_ROOT = "./data"

def load_data() -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict]:
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    batch_size = 4
    if training_set == 'CIFAR':
        trainset = CIFAR10(DATA_ROOT, train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
        testset = CIFAR10(DATA_ROOT, train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

    elif training_set == 'product':
        trainset = product_dataset.image_datasets
        trainloader = torch.utils.data.DataLoader(trainset[folder], batch_size=batch_size, shuffle=True, drop_last=True)
        testset = product_dataset.image_datasets
        testloader = torch.utils.data.DataLoader(trainset[val_folder], batch_size=batch_size, shuffle=True, drop_last=True)
        
    num_examples = {"trainset" : len(trainset[folder]), "testset" : len(testset[val_folder])}
    print(num_examples)
    return trainloader, testloader, num_examples

def train(
    net: Net,
    trainloader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device,
) -> None:
    """Train the network."""
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")
    # Train the network
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        con = 0

        for i, data in enumerate(trainloader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            con+=1
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            print(con, running_loss/con) 

            if i % 100 == 99:  # print every 100 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0


def test(
    net: Net,
    testloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate the network on the entire test set."""
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print("Accuracy of the network on the 10000 test images: %d %%" % (100 * accuracy))
    return loss, accuracy


def main():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Centralized PyTorch training")
    print("Load data")
    trainloader, testloader, _ = load_data()
    print("Start training")
    net=Net().to(DEVICE)
    train(net=net, trainloader=trainloader, epochs=2, device=DEVICE)
    print("Evaluate model")
    loss, accuracy = test(net=net, testloader=testloader, device=DEVICE)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)


if __name__ == "__main__":
    main()