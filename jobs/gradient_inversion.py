# !pip install torchvision matplotlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ===== Step 1: Prepare MNIST dataset =====
transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)

# ===== Step 2: Define a simple model =====
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
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

# Save the true gradients (as if they were sent to a server in FL)
original_grads = [p.grad.clone() for p in model.parameters()]

# ===== Step 4: Attack: reconstruct the image from gradients =====
dummy_data = torch.randn_like(images, requires_grad=True)
dummy_label = torch.randn((1, 10), requires_grad=True)

optimizer = optim.LBFGS([dummy_data, dummy_label], lr=1)

history = []
for i in range(250):
    def closure():
        optimizer.zero_grad()
        pred = model(dummy_data)
        dummy_loss = F.cross_entropy(pred, dummy_label.softmax(dim=-1))
        grads = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

        grad_diff = 0
        for g1, g2 in zip(grads, original_grads):
            grad_diff += ((g1 - g2)**2).sum()

        grad_diff.backward()
        return grad_diff

    optimizer.step(closure)

    if i % 50 == 0:
        history.append(dummy_data.detach().clone())

# ===== Step 5: Visualize reconstruction =====
fig, axes = plt.subplots(1, len(history)+1, figsize=(10, 3))
axes[0].imshow(images[0].squeeze(), cmap='gray')
axes[0].set_title("Original")
axes[0].axis('off')

for idx, img in enumerate(history):
    axes[idx+1].imshow(img[0].squeeze(), cmap='gray')
    axes[idx+1].set_title(f"Iter {idx*50}")
    axes[idx+1].axis('off')

plt.show()
