# utilities/train_model.py

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch
import torchvision
from torchvision import datasets, transforms
import timm
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from PIL import Image
from tqdm import tqdm

# Swin Transformer classifier using timm
class CustomSwinTransformer(nn.Module):
    def __init__(self, pretrained=True, num_classes=7):
        super().__init__()
        self.backbone = timm.create_model(
            'swin_base_patch4_window7_224', pretrained=pretrained, num_classes=0
        )
        self.classifier = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)

# Training function
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data)
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct.double() / total
    return epoch_loss, epoch_acc.item()

if __name__ == "__main__":
    print("Starting training script...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root="FER2013_processed/train", transform=transform)
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Classes found: {train_dataset.classes}")
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    model = CustomSwinTransformer(pretrained=True, num_classes=7).to('cuda')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(5):
        print(f"Starting epoch {epoch + 1}")
        loss, acc = train_one_epoch(model, train_loader, optimizer, criterion, 'cuda')
        print(f"Epoch {epoch + 1}: Loss={loss:.4f}, Accuracy={acc:.4f}")
    print("Training script finished.")
    # Save trained model weights
    torch.save(model.state_dict(), "best_model.pth")
    print("Model weights saved to best_model.pth.")
