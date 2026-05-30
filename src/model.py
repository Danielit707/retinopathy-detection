# src/model.py
import torch
import torch.nn as nn
import torchvision.models as models

class RetinopathyEfficientNet(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(RetinopathyEfficientNet, self).__init__()
        
        # Load the pre-trained EfficientNet-B0 backbone
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        
        # Freeze the feature extractor parameters to preserve Transfer Learning features
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Get the input feature count for the original classifier
        in_features = self.backbone.classifier[1].in_features
        
        # Custom Multi-Layer Perceptron classifier head
        self.backbone.classifier[1] = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)