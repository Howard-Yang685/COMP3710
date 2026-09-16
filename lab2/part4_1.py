import argparse
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid


class OASISDataset(Dataset):
    def __init__(self, directory, image_size):
        self.paths = sorted(Path(directory).glob("*.png"))
        if not self.paths:
            raise RuntimeError(f"No PNG images found in {directory}")
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("L")
        return self.transform(image)


class VAE(nn.Module):
    def __init__(self, image_size=128, latent_dim=2):
        super().__init__()
        if image_size % 16 != 0:
            raise ValueError("image_size must be divisible by 16")

        self.image_size = image_size
        self.latent_dim = latent_dim
        self.feature_size = image_size // 16

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(inplace=True),
        )

        feature_count = 256 * self.feature_size * self.feature_size
        self.fc_mu = nn.Linear(feature_count, latent_dim)
        self.fc_logvar = nn.Linear(feature_count, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, feature_count)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        features = self.encoder(x).flatten(start_dim=1)
        return self.fc_mu(features), self.fc_logvar(features)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def decode(self, z):
        features = self.decoder_input(z)
        features = features.view(-1, 256, self.feature_size, self.feature_size)
        return self.decoder(features)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(reconstruction, image, mu, logvar, beta):
    reconstruction_loss = F.binary_cross_entropy(
        reconstruction, image, reduction="sum"
    )
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + beta * kl_loss, reconstruction_loss, kl_loss


def run_epoch(model, loader, device, beta, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = np.zeros(3, dtype=np.float64)

    for images in loader:
        images = images.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            reconstruction, mu, logvar = model(images)
            loss, reconstruction_loss, kl_loss = vae_loss(
                reconstruction, images, mu, logvar, beta
            )
            if training:
                loss.backward()
                optimizer.step()

        totals += np.array(
            [loss.item(), reconstruction_loss.item(), kl_loss.item()]
        )

    return tuple(totals / len(loader.dataset))


def save_reconstructions(model, loader, device, output_path):
    model.eval()
    images = next(iter(loader))[:8].to(device)
    with torch.no_grad():
        reconstructions, _, _ = model(images)

    comparison = torch.cat([images.cpu(), reconstructions.cpu()])
    grid = make_grid(comparison, nrow=8, padding=2, pad_value=1.0)
    plt.figure(figsize=(16, 4))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.axis("off")
    plt.title("Original images (top) and reconstructions (bottom)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def save_manifold(model, device, output_path, grid_size=15):
    model.eval()
    values = torch.linspace(-3.0, 3.0, grid_size)
    latent_points = torch.tensor(
        [[x.item(), y.item()] for y in reversed(values) for x in values],
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        generated = model.decode(latent_points).cpu()

    grid = make_grid(generated, nrow=grid_size, padding=1, pad_value=1.0)
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.axis("off")
    plt.title("VAE manifold sampled from the 2D latent space")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def save_loss_curve(train_losses, validation_losses, output_path):
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Training loss")
    plt.plot(epochs, validation_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss per image")
    plt.title("VAE training history")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train a VAE on OASIS MRI slices")
    parser.add_argument(
        "--data-root",
        default="/home/groups/comp3710/OASIS",
    )
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("A CUDA GPU is required. Submit this script through Slurm.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = OASISDataset(
        Path(args.data_root) / "keras_png_slices_train", args.image_size
    )
    validation_dataset = OASISDataset(
        Path(args.data_root) / "keras_png_slices_validate", args.image_size
    )
    test_dataset = OASISDataset(
        Path(args.data_root) / "keras_png_slices_test", args.image_size
    )

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": True,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, **loader_options
    )
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

    model = VAE(args.image_size, args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Dataset sizes: train={len(train_dataset)}, "
        f"validation={len(validation_dataset)}, test={len(test_dataset)}"
    )
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_losses = []
    validation_losses = []
    best_validation_loss = float("inf")
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_reconstruction, train_kl = run_epoch(
            model, train_loader, device, args.beta, optimizer
        )
        validation_loss, _, _ = run_epoch(
            model, validation_loader, device, args.beta
        )
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(model.state_dict(), output_dir / "vae_model.pt")

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train={train_loss:.3f} | recon={train_reconstruction:.3f} | "
            f"kl={train_kl:.3f} | validation={validation_loss:.3f}"
        )

    elapsed = time.perf_counter() - start_time
    model.load_state_dict(
        torch.load(output_dir / "vae_model.pt", map_location=device, weights_only=True)
    )
    test_loss, test_reconstruction, test_kl = run_epoch(
        model, test_loader, device, args.beta
    )

    save_loss_curve(
        train_losses, validation_losses, output_dir / "loss_curve.png"
    )
    save_reconstructions(
        model, test_loader, device, output_dir / "reconstruction.png"
    )
    save_manifold(model, device, output_dir / "manifold.png")

    print(f"Training time: {elapsed:.2f} seconds")
    print(
        f"Test loss: {test_loss:.3f} | "
        f"reconstruction={test_reconstruction:.3f} | kl={test_kl:.3f}"
    )
    print(f"Results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
