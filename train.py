"""
DeepVision — Training Script
Author: Shebin S Illikkal | Shebinsillikkal@gmail.com
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from models.classifier import DeepVisionClassifier
import wandb, os, json
from pathlib import Path

def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

def train(config: dict):
    wandb.init(project="deepvision", config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    dataset = datasets.ImageFolder(config['data_dir'], transform=get_transforms(True))
    val_size = int(0.2 * len(dataset))
    train_set, val_set = random_split(dataset, [len(dataset) - val_size, val_size])
    val_set.dataset.transform = get_transforms(False)

    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=config['batch_size'], num_workers=4)

    model = DeepVisionClassifier(num_classes=len(dataset.classes), pretrained=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(config['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        wandb.log({'train_loss': train_loss, 'train_acc': train_acc,
                   'val_loss': val_loss, 'val_acc': val_acc, 'epoch': epoch})
        print(f"Epoch {epoch+1}/{config['epochs']} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  ✓ Saved best model (acc={best_acc:.4f})")

    print(f"Training complete. Best val accuracy: {best_acc:.4f}")
    wandb.finish()

if __name__ == "__main__":
    config = {
        "data_dir": os.getenv("DATA_DIR", "./data"),
        "epochs": int(os.getenv("EPOCHS", "30")),
        "batch_size": int(os.getenv("BATCH_SIZE", "32")),
        "lr": float(os.getenv("LR", "0.0003")),
        "num_classes": int(os.getenv("NUM_CLASSES", "2"))
    }
    train(config)
