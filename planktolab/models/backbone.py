from torchvision import models
import torch.nn as nn
import torch


model_names = ["resnet34", "efficientnet_b0", "mobilenet_v3", "mobilenet_v3_large"]


def create_model(model_name, num_classes):
    if model_name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "mobilenet_v3":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    
    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    elif model_name == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


class ClassificationModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()

        if model_name == "resnet34":
            backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()

        elif model_name == "efficientnet_b0":
            backbone = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT
            )
            in_features = backbone.classifier[1].in_features
            backbone.classifier[1] = nn.Identity()

        elif model_name == "mobilenet_v3":
            backbone = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.DEFAULT
            )
            in_features = backbone.classifier[3].in_features
            backbone.classifier[3] = nn.Identity()

        elif model_name == "mobilenet_v3_large":
            backbone = models.mobilenet_v3_large(
                weights=models.MobileNet_V3_Large_Weights.DEFAULT
            )
            in_features = backbone.classifier[3].in_features
            backbone.classifier[3] = nn.Identity()

        elif model_name == "convnext_tiny":
            backbone = models.convnext_tiny(
                weights=models.ConvNeXt_Tiny_Weights.DEFAULT
            )
            in_features = backbone.classifier[2].in_features
            backbone.classifier[2] = nn.Identity()

        else:
            raise ValueError(f"Unknown model: {model_name}")

        self.backbone = backbone
        self.head = nn.Linear(in_features, num_classes)

    def forward_features(self, x):
        return self.backbone(x)

    def forward(self, x):
        features = self.forward_features(x)
        return self.head(features)


def get_available_models():
    return model_names


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model