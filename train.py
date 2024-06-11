import os
from dataset_loader import get_datasets
from model import EyeFixationNetwork
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from tqdm import tqdm


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)

# Dataset load
train_dataset, val_dataset, _ = get_datasets()
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)


model = EyeFixationNetwork()
model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

os.makedirs("final_model_pth", exist_ok=True)

patience = 2  
best_val_loss = np.Inf
counter = 0
num_epochs = 20

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):


        inputs = batch["image"].to(device)
        labels = batch["fixation"].to(device)

        optimizer.zero_grad()
        pred = model(inputs)
        
        labels = F.interpolate(labels, size=(224, 224), mode='nearest')
        loss = criterion(pred, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in val_dataloader:
            inputs = batch["image"].to(device)
            labels = batch["fixation"].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_epoch_loss = val_loss / len(val_dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_epoch_loss:.4f}")

    # Early stopping
    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        counter = 0
        torch.save(model.state_dict(), f"final_model_pth/model_epoch_{epoch + 1}.pth")
    else:
        counter += 1
        if counter >= patience:
            print(f"Validation loss did not improve for {patience} epochs. Early stopping...")
            break
