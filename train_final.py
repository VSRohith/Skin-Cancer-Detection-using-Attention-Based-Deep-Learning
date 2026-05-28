import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from data_loader import SkinDataset, train_transform, val_transform
from model_attention import SkinAttentionModel
from tqdm import tqdm

# --- 1. CONFIGURATION ---
CSV_PATH = r"D:\CLG DOCS\Mini-project\Method1-code\Skin_Cancer_Project\ham1000_metadata.csv"
IMG_DIR = r"D:\CLG DOCS\Mini-project\Method1-code\Skin_Cancer_Project\data"
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. DATA PREPARATION & BALANCING ---
df = pd.read_csv(CSV_PATH)

# Map diagnosis to numeric IDs for sampling
unique_labels = sorted(df['dx'].unique())
label_to_id = {label: i for i, label in enumerate(unique_labels)}
df['label_id'] = df['dx'].map(label_to_id)

# Split data
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['dx'])

# Calculate weights for balancing classes
class_counts = train_df['label_id'].value_counts().sort_index().values
class_weights = 1. / class_counts
sample_weights = [class_weights[t] for t in train_df['label_id']]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# --- 3. DATALOADERS ---
train_dataset = SkinDataset(train_df, IMG_DIR, transform=train_transform)
val_dataset = SkinDataset(val_df, IMG_DIR, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 4. MODEL SETUP ---
model = SkinAttentionModel(num_classes=7).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

# --- 5. TRAINING ENGINE ---
print(f"🚀 Training starting on {DEVICE}...")
best_val_acc = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
    
    for images, labels in train_loop:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_loop.set_postfix(loss=loss.item())

    # Validation Phase
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    scheduler.step(val_acc)
    
    print(f"📊 Epoch {epoch+1} Summary: Val Acc: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']}")

    # Save the best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'skin_attention_model.pth')
        print(f"⭐ New Best Model Saved! ({val_acc:.2f}%)")

print("\n✨ Retraining Complete. Use the new .pth file in your app.")