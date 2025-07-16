from basemodel import BaseModel
import torch
from torch import nn
import timm
import torchvision.transforms as transforms


class VisionTransformer(BaseModel, nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("vit_base_patch16_224.mae", pretrained=False)
        self.backbone.head = nn.Linear(768, 1)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def load(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.load_state_dict(state_dict)
        self.eval()

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = self.transform(x).unsqueeze(0)
            logits = self.forward(x)
            return torch.sigmoid(logits).item()
