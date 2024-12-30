import torch
import torchvision.models as models

def load_model():
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove last layer
    model.eval()
    return model
