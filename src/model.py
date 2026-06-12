import torch
import torch.nn as nn
import torchvision.models as models

class RetinopathyEfficientNet(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(RetinopathyEfficientNet, self).__init__()
        
        # Load pre-trained EfficientNet-B0 backbone
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        
        # Freeze early layers to preserve low-level feature extractors (edges, circles)
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Unfreeze deep feature blocks (Blocks 6 and 7) for targeted medical domain fine-tuning
        for param in self.backbone.features[6:].parameters():
            param.requires_grad = True
            
        in_features = self.backbone.classifier[1].in_features
        
        # Upgraded MLP classifier head using SiLU (Swish) to match EfficientNet architecture style
        self.backbone.classifier[1] = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.SiLU(), 
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)