import torch
import torch.nn as nn
import torchvision.models as models
#from torchvision.models import ResNet18_Weights




class ResNet18(nn.Module):
    def __init__(self, num_classes, dropout=0.5):
        super(ResNet18, self).__init__()
        # Load pre-trained ResNet18 with Default weights
        self.resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Remove the original fully connected layer to customize the network
        # This allows us to add our own classification head
        self.resnet18.fc = nn.Identity()
        
        # Custom classification layers for feature refinement and final classification
        # Designed to work with 64x64 input images after ResNet18 feature extraction
        self.custom_layers = nn.Sequential(
            # Reduce feature complexity while maintaining spatial information
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),  # Maintains 8x8 spatial dimensions
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),  # Changed from Dropout2d to Dropout
            
            # Further feature reduction
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),  # Maintains 8x8 spatial dimensions
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            # Final feature reduction before classification
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),  # Maintains 8x8 spatial dimensions
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            # Global average pooling to reduce spatial dimensions to 1x1
            nn.AdaptiveAvgPool2d((1, 1)),
            
            # Flatten the output for fully connected layer
            nn.Flatten(),
            
            # Final classification layer with specified number of classes
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # Forward pass through ResNet18 feature extraction layers
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)
        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)
        
        # Pass through custom classification layers
        x = self.custom_layers(x)
        return x