from basemodel import BaseModel
import torch
from torch import nn
import timm
import torchvision.transforms as transforms


class EfficientNet(BaseModel, nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=False)
        self.backbone.classifier = nn.Linear(1280, 1)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def load(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.load_state_dict(state_dict)
        self.eval()

    def predict(self, image):
        self.eval()
        with torch.no_grad():
            x = self.transform(image).unsqueeze(0)
            logits = self.backbone(x)
            return torch.sigmoid(logits).item()
