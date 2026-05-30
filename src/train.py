# src/train.py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from dataset import get_data_loaders
from model import RetinopathyEfficientNet

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass using Mixed Precision to optimize VRAM
        with torch.amp.autocast(device_type="cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        # Backward pass with scaled gradients
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Metrics tracking
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data)
        total_samples += images.size(0)
        
    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions.double() / total_samples
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += images.size(0)
            
    val_loss = running_loss / total_samples
    val_acc = correct_predictions.double() / total_samples
    return val_loss, val_acc

def main():
    # Hyperparameters
    BATCH_SIZE = 16
    EPOCHS = 5
    LEARNING_RATE = 0.001
    
    # Path configurations relative to project root
    TRAIN_CSV = os.path.join("data", "train.csv")
    VAL_CSV = os.path.join("data", "val.csv")
    IMG_DIR = os.path.join("data", "raw")
    WEIGHTS_PATH = os.path.join("weights", "best_efficientnet.pth")
    
    # Runtime Hardware Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        
    # Balanced loss setup to combat severe class distribution skewness
    print("Calculating class weights to counter dataset imbalance...")
    train_df = pd.read_csv(TRAIN_CSV)
    y_train = train_df['diagnosis'].values
    unique_classes = np.unique(y_train)
    
    calculated_weights = compute_class_weight(
        class_weight='balanced', 
        classes=unique_classes, 
        y=y_train
    )
    class_weights = torch.tensor(calculated_weights, dtype=torch.float).to(device)
    print(f"Class weights strictly enforced: {calculated_weights}")
        
    # Get optimized PyTorch DataLoaders
    train_loader, val_loader = get_data_loaders(
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        img_dir=IMG_DIR,
        batch_size=BATCH_SIZE
    )
    
    # Initialization
    model = RetinopathyEfficientNet(num_classes=5, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights) # Injected penalization weights
    optimizer = optim.Adam(model.backbone.classifier.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda") # Positional setup to avoid syntax mismatches
    
    best_acc = 0.0
    
    print("\n--- Starting Training Pipeline ---")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
              
        # Keep track and save optimal weights checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), WEIGHTS_PATH)
            print(f"-> Saved new best model checkpoint to {WEIGHTS_PATH}")
            
    print("\nTraining test pipeline completed successfully!")

if __name__ == "__main__":
    main()