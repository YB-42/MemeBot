import timm
import torch
import torchvision.transforms as transforms
from basemodel import BaseModel
from torch import nn


class ResNet(BaseModel, nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("resnet50.a1_in1k", pretrained=False)
        self.backbone.fc = nn.Linear(2048, 1)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def load(self, checkpoint_path: str) -> None:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.load_state_dict(state_dict)
        self.eval()

    def predict(self, x) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            x = self.transform(x).unsqueeze(0)
            logits = self.forward(x)
            return torch.sigmoid(logits).item()
