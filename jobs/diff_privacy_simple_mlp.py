import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity


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

def clip_gradients(parameters, max_norm):
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)

    return total_norm


def add_dp_noise(parameters, max_norm, noise_multiplier):
    for p in parameters:
        if p.grad is not None:
            noise = torch.normal(
                mean=0,
                std=noise_multiplier * max_norm,
                size=p.grad.shape
            )
            p.grad.data.add_(noise)



def train():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)

    model = SimpleMLP()

    images, labels = next(iter(trainloader))
    criterion = nn.CrossEntropyLoss()

    model.zero_grad()
    output = model(images)
    loss = criterion(output, labels)
    loss.backward()

    # ===== DP parameters =====
    MAX_NORM = 1.0  # C
    NOISE_MULTIPLIER = 1.0  # σ

    # Something wrong here

    grad_norm = clip_gradients(model.parameters(), MAX_NORM)

    add_dp_noise(model.parameters(), MAX_NORM, NOISE_MULTIPLIER)

    grads = [p.grad.clone() for p in model.parameters()]

    # Something wrong here

    return {
        "model": model,
        "model_name": "SimpleMLP_with_noise",
        "manifest": {
            "framework": "pytorch",
            "architecture": "SimpleMLP",
            "serialization_type": "state_dict"
        },
        "artifacts": {
            "gradient": grads
        }
    }