import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
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
        
        with torch.amp.autocast(device_type="cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
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

def run_fold(fold_idx, device, img_dir, batch_size, epochs, lr):
    print(f"\n=================== 🏁 STARTING FOLD [{fold_idx}/5] ===================")
    
    train_csv = os.path.join("data", "folds", f"train_fold_{fold_idx}.csv")
    val_csv = os.path.join("data", "folds", f"val_fold_{fold_idx}.csv")
    weights_path = os.path.join("weights", f"best_efficientnet_fold_{fold_idx}.pth")
    
    train_df = pd.read_csv(train_csv)
    y_train = train_df['diagnosis'].values
    unique_classes = np.unique(y_train)
    calculated_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_train)
    class_weights = torch.tensor(calculated_weights, dtype=torch.float).to(device)
    
    train_loader, val_loader = get_data_loaders(
        train_csv=train_csv, val_csv=val_csv, img_dir=img_dir, batch_size=batch_size
    )
    
    model = RetinopathyEfficientNet(num_classes=5, pretrained=True).to(device)
    
    # CrossEntropy upgraded with label smoothing to deal with highly subjective bordering stages
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    # Upgraded to AdamW for superior weight decay mechanics during backpropagation fine-tuning
    optimizer = optim.AdamW([
        {'params': model.backbone.features[6:].parameters(), 'lr': lr * 0.1},
        {'params': model.backbone.classifier.parameters(), 'lr': lr}
    ], weight_decay=1e-4)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda")
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        current_lr = optimizer.param_groups[1]['lr']
        
        print(f"Fold {fold_idx} | Epoch [{epoch+1}/{epochs}] | Head LR: {current_lr:.6f} -> "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
              
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), weights_path)
            print(f"   -> 🔥 Saved new best model checkpoint for Fold {fold_idx} to {weights_path}")
            
    print(f"=================== 🟥 FINISHED FOLD [{fold_idx}/5] Best Val Acc: {best_acc:.4f} ===================\n")
    return best_acc

def main():
    BATCH_SIZE = 16
    EPOCHS = 15
    LEARNING_RATE = 0.001
    NUM_FOLDS = 5
    
    IMG_DIR = os.path.join("data", "raw")
    os.makedirs("weights", exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using architectural hardware device: {device}")
    if device.type == "cuda":
        print(f"GPU Node Name: {torch.cuda.get_device_name(0)}")
        
    fold_scores = []
    
    for fold in range(NUM_FOLDS):
        best_fold_acc = run_fold(
            fold_idx=fold, 
            device=device, 
            img_dir=IMG_DIR, 
            batch_size=BATCH_SIZE, 
            epochs=EPOCHS, 
            lr=LEARNING_RATE
        )
        fold_scores.append(best_fold_acc.item() if torch.is_tensor(best_fold_acc) else best_fold_acc)
        
    print("\n--- 🏆 CROSS-VALIDATION PIPELINE COMPLETE 🏆 ---")
    print(f"All fold accuracies: {fold_scores}")
    print(f"Mean CV Accuracy: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")

if __name__ == "__main__":
    main()