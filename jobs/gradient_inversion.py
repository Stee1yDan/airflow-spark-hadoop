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

    original_grads = torch.load(ctx.artifacts_dir / "artifacts/gradient.pt", map_location="cpu")
    original_image = np.load(ctx.artifacts_dir / "artifacts/original_image.npy")

    # ===== Step 4: Attack: reconstruct the image from gradients =====
    cos_sim = 0
    cos_sim_score = 0.1
    for i in range(1, 10):
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
    orig = original_image
    recon = dummy_data.detach().cpu().numpy()[0, 0]
    recon = np.nan_to_num(recon, nan=0.0)
    orig = np.nan_to_num(orig, nan=0.0)

    mse = float(np.mean((orig - recon) ** 2))
    cos_sim = float(
        cosine_similarity(orig.reshape(1, -1), recon.reshape(1, -1))[0][0]
    )
    ssim_score = ssim(
        orig, recon, data_range=orig.max() - recon.min()
    )
    attribute_accuracy = int(torch.argmax(model(dummy_data)).item() == labels.item())

    orig_np = np.array(orig, dtype=np.float32)
    recon_np = np.array(recon, dtype=np.float32)

    r_cos = max(0.0, min(1.0, cos_sim))
    r_ssim = max(0.0, min(1.0, ssim_score))
    r_attr = float(attribute_accuracy)
    r_mse = 1.0 / (1.0 + mse)

    weights = {
        "mse": 0.2,
        "cosine": 0.6,
        "ssim": 0.2,
        "attribute": 0
    }
    weight_sum = sum(weights.values())
    weights = {k: v / weight_sum for k, v in weights.items()}

    R_scenario = (
        weights["mse"] * r_mse
        + weights["cosine"] * r_cos
        + weights["ssim"] * r_ssim
        + weights["attribute"] * r_attr
    )

    metrics = {
        "mse": mse,
        "cosine_similarity": cos_sim,
        "ssim": ssim_score,
        "attribute_accuracy": attribute_accuracy,
    }
    metrics.update({
        "r_mse": r_mse,
        "r_cosine": r_cos,
        "r_ssim": r_ssim,
        "r_attribute": r_attr,
        "R_scenario": R_scenario
    })

    # ===================================================================
    # UPDATED: Return matplotlib figures instead of raw numpy arrays
    # The test runner you already have will automatically detect plt.Figure
    # objects and save them as clean, high-quality PNGs with titles, etc.
    # ===================================================================
    # Ensure correct 28x28 shape (in case the arrays are flattened)
    orig_img = orig_np.reshape(28, 28) if orig_np.ndim == 1 else orig_np
    recon_img = recon_np.reshape(28, 28) if recon_np.ndim == 1 else recon_np

    # Original image figure
    fig_original = plt.figure(figsize=(6, 6))
    plt.imshow(orig_img, cmap="gray")
    plt.title("Original MNIST Image", fontsize=14)
    plt.axis("off")

    # Reconstructed image figure
    fig_reconstruction = plt.figure(figsize=(6, 6))
    plt.imshow(recon_img, cmap="gray")
    plt.title("Reconstructed Image\n(Gradient Inversion Attack)", fontsize=14)
    plt.axis("off")

    artifacts = {
        "original": fig_original,
        "reconstruction": fig_reconstruction
    }

    return {"metrics": metrics, "artifacts": artifacts}