import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=4, pretrained=True):
    if pretrained:
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
    else:
        model = models.resnet50()

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model