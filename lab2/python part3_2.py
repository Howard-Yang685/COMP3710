import time
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Settings
MODE = "train"  # "train", "test", or "demo"
CHECKPOINT = Path("cifar10_best.pt")
SEED, EPOCHS, BATCH_SIZE, NUM_WORKERS = 42, 50, 512, 4
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amp = device.type == "cuda"
if amp:
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
print("Device:", torch.cuda.get_device_name(0) if amp else device, "| AMP:", amp)

# Data
mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(2, 9),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    transforms.RandomErasing(p=0.25)
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
train_data = datasets.CIFAR10("data", train=True, download=True, transform=train_transform)
val_data = datasets.CIFAR10("data", train=True, transform=test_transform)
test_data = datasets.CIFAR10("data", train=False, download=True, transform=test_transform)
generator = torch.Generator().manual_seed(SEED)
indices = torch.randperm(50000, generator=generator).tolist()
train_ids, val_ids = indices[:45000], indices[45000:]

def loader(dataset, shuffle=False):
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=amp,
                      persistent_workers=NUM_WORKERS > 0)

train_loader = loader(Subset(train_data, train_ids), True)
val_loader = loader(Subset(val_data, val_ids))
test_loader = loader(test_data)

# ResNet-18
def conv_bn(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_channels)
    )

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.main = nn.Sequential(
            conv_bn(in_channels, out_channels, stride),
            nn.ReLU(),
            conv_bn(out_channels, out_channels)
        )
        self.shortcut = nn.Identity() if stride == 1 and in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return torch.relu(self.main(x) + self.shortcut(x))

class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [conv_bn(3, 64), nn.ReLU()]
        channels = 64
        for width, stride in ((64, 1), (128, 2), (256, 2), (512, 2)):
            layers += [BasicBlock(channels, width, stride), BasicBlock(width, width)]
            channels = width
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        return self.classifier(self.features(x))

model = ResNet18().to(device)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

def train_epoch(optimizer, scaler, scheduler=None):
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            loss = loss_fn(model(images), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler:
            scheduler.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

@torch.no_grad()
def accuracy(data_loader, tta=False):
    model.eval()
    correct, total = 0, 0
    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(images)
            if tta:
                logits = (logits + model(images.flip(-1))) / 2
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)
    return correct / total

# Train or load
if MODE == "train":
    optimizer = torch.optim.SGD(model.parameters(), lr=0.04, momentum=0.9,
                                weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.4, epochs=EPOCHS, steps_per_epoch=len(train_loader)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best_accuracy = 0.0
    if amp:
        torch.cuda.synchronize()
    start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(optimizer, scaler, scheduler)
        val_accuracy = accuracy(val_loader)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), CHECKPOINT)
        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch}/{EPOCHS} | Loss {loss:.4f} | Val {val_accuracy:.2%}")
    if amp:
        torch.cuda.synchronize()
    training_time = time.perf_counter() - start
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    print(f"Training time: {training_time:.2f}s")
    print(f"Best validation: {best_accuracy:.2%}")
else:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError("Train the model or set CHECKPOINT.")
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))

# Final test
test_accuracy = accuracy(test_loader, tta=True)
print(f"Test accuracy: {test_accuracy:.2%}")
print("Accuracy >= 94%:", test_accuracy >= 0.94)
if MODE == "train":
    print("Time <= 360s:", training_time <= 360)

# Cluster demonstration
if MODE == "demo":
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    if amp:
        torch.cuda.synchronize()
    start = time.perf_counter()
    loss = train_epoch(optimizer, scaler)
    if amp:
        torch.cuda.synchronize()
    print(f"Demo epoch | Loss {loss:.4f} | Time {time.perf_counter() - start:.2f}s")