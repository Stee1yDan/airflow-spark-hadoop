import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity

def run():
    # ===== Step 1: Prepare MNIST dataset =====
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)

    # ===== Step 2: Define a simple model =====
    class SimpleMLP(nn.Module):
        def __init__(self):
            super(SimpleMLP, self).__init__()
            self.fc1 = nn.Linear(28 * 28, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = x.view(-1, 28 * 28)
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    model = SimpleMLP()

    # ===== Step 3: Simulate a real training step to get "true gradients" =====
    images, labels = next(iter(trainloader))
    criterion = nn.CrossEntropyLoss()

    model.zero_grad()
    output = model(images)
    loss = criterion(output, labels)
    loss.backward()

    result = {
        "model": model
    }

    return result