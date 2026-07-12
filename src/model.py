import torch
import torch.nn as nn
import torchvision.models as models

class AgeGenderModel(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=True):
        super(AgeGenderModel, self).__init__()
        
        if backbone == "resnet18":
            try:
                weights = models.ResNet18_Weights.DEFAULT if pretrained else None
                self.backbone = models.resnet18(weights=weights)
            except AttributeError:
                self.backbone = models.resnet18(pretrained=pretrained)
                
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif backbone == "mobilenet_v3_small":
            try:
                weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
                self.backbone = models.mobilenet_v3_small(weights=weights)
            except AttributeError:
                self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
                
            # MobileNetV3 uses a multi-layer classifier head
            # We fetch its input dimension, then override it
            in_features = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
            
        # Age head: regression output for age (continuous variable)
        self.age_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Gender head: 2-class classification (0: Male, 1: Female)
        self.gender_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        age_out = self.age_head(features).squeeze(-1)  # (batch_size,)
        gender_out = self.gender_head(features)        # (batch_size, 2)
        return age_out, gender_out
