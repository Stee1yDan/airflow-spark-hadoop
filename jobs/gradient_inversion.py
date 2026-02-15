# !pip install torchvision matplotlib

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from typing import Any, Dict
from pathlib import Path

from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity

from dataclasses import dataclass

@dataclass
class TestContext:
    model_name: Any | None
    model_path: str | None
    artifacts_dir: Path
    metadata: Dict[str, Any]
    local_model_path: Path | None


def run(ctx: TestContext):
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
    state = torch.load(ctx.local_model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # ===== Step 1: Prepare MNIST dataset =====
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)


    # ===== Step 3: Simulate a real training step to get "true gradients" =====
    images, labels = next(iter(trainloader))
    criterion = nn.CrossEntropyLoss()

    model.zero_grad()
    output = model(images)
    loss = criterion(output, labels)
    loss.backward()

    # Save the true gradients (as if they were sent to a server in FL)
    original_grads = torch.load(ctx.artifacts_dir / "artifacts/gradient.pt", map_location="cpu")

    # ===== Step 4: Attack: reconstruct the image from gradients =====
    cos_sim = 0
    cos_sim_score = 1

    while cos_sim < cos_sim_score:
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
                    grad_diff += ((g1 - g2) ** 2).sum()

                grad_diff.backward()
                return grad_diff

            optimizer.step(closure)

            if i % 50 == 0:
                history.append(dummy_data.detach().clone())

        # ===== Step 5: Metrics =====

        all_images = torch.stack([img for img, lbl in trainset])  # shape: [N, 1, 28, 28]
        all_images_flat = all_images.view(all_images.size(0), -1)  # [N, 784]

        recon_flat = dummy_data.detach().cpu().view(-1)  # [784]

        mse_distances = torch.mean((all_images_flat - recon_flat) ** 2, dim=1)  # [N]
        closest_idx = torch.argmin(mse_distances)
        orig = all_images[closest_idx, 0].cpu().numpy()  # single 28x28 image

        recon = dummy_data.detach().cpu().numpy()[0, 0]
        recon = np.nan_to_num(recon, nan=0.0)
        orig = np.nan_to_num(orig, nan=0.0)

        mse = float(np.mean((orig - recon) ** 2))
        cos_sim = float(
            cosine_similarity(orig.reshape(1, -1), recon.reshape(1, -1))[0][0]
        )
        ssim_score = ssim(
            orig,
            recon,
            data_range=orig.max() - recon.min()
        )

        attribute_accuracy = int(torch.argmax(model(dummy_data)).item() == labels.item())

        orig_np = np.array(orig, dtype=np.float32)
        recon_np = np.array(recon, dtype=np.float32)

        cos_sim_score -= 0.05


    metrics = {
        "mse": mse,
        "cosine_similarity": cos_sim,
        "ssim": ssim_score,
        "attribute_accuracy": attribute_accuracy,
    }

    artifacts = {
        "original": orig_np,
        "reconstruction": recon_np
    }


    return {"metrics": metrics, "artifacts": artifacts}