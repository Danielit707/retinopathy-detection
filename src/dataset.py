# src/dataset.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class APTOSDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # APTOS dataset uses 'id_code' without extension as filename
        img_name = os.path.join(self.img_dir, f"{self.df.iloc[idx, 0]}.png")
        image = Image.open(img_name).convert('RGB')
        
        # Diagnosis target (classes 0 to 4)
        label = int(self.df.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_data_loaders(train_csv, val_csv, img_dir, batch_size=16, num_workers=2):
    """
    Helper function to generate optimized training and validation DataLoaders.
    """
    # ImageNet normalization standard transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1), # Account for camera flash variations
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Instantiating the datasets
    train_dataset = APTOSDataset(csv_file=train_csv, img_dir=img_dir, transform=train_transform)
    val_dataset = APTOSDataset(csv_file=val_csv, img_dir=img_dir, transform=val_transform)

    # pin_memory=True speeds up RAM to GPU data transfer transfers
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
