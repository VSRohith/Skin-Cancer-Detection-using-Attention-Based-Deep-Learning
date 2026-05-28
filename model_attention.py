import torch
import torch.nn as nn
import torchvision.models as models
from attention_blocks import ChannelAttention, SpatialAttention

class SkinAttentionModel(nn.Module):
    def __init__(self, num_classes=7):
        super(SkinAttentionModel, self).__init__()
        
        # 1. The Body: Using ResNet50
        resnet = models.resnet50(weights='DEFAULT')
        # Remove the last two layers (avgpool and fc)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # 2. CBAM Attention Mechanism
        self.ca = ChannelAttention(2048)
        self.sa = SpatialAttention()
        
        # 3. Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.features(x)   # Shape: [batch, 2048, 7, 7]
        x = self.ca(x) * x     # Channel Attention
        x = self.sa(x) * x     # Spatial Attention
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x