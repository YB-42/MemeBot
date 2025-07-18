import torch
from models.effnet import EfficientNet
from PIL import Image
from models.resnet import ResNet
from models.vit import VisionTransformer


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

    # Заменяем на переданное изображение, а не жестко "image.jpg"
    # image_filt = Image.open("image.jpg").convert("RGB")
    image_filt = image.convert("RGB")

    prediction = model.predict(image_filt)

    return prediction


if __name__ == "__main__":
    model_n = input("Введите название модели: ")
    image = Image.open("image.jpg")
    print(f"Prediction: {run_inference(model_n, image)}")
