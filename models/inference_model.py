import torch
from effnet import EfficientNet
from PIL import Image
from resnet import ResNet
from vit import VisionTransformer


def run_inference(model_name: str, image: Image.Image) -> torch.Tensor:
    if model_name == "EfficientNet":
        model = EfficientNet()
        model.load("data/EffNet.pth")
    elif model_name == "ResNet":
        model = ResNet()
        model.load("data/ResNet.pth")
    elif model_name == "ViT":
        model = VisionTransformer()
        model.load("data/ViT.pth")
    else:
        raise ValueError("Invalid model name")

    image_filt = Image.open("image.jpg").convert("RGB")
    prediction = model.predict(image_filt)

    return prediction


model_n = input()
image = Image.open("image.jpg")
print(f"Prediction: {run_inference(model_n, image)}")

