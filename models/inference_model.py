import torch
from PIL import Image

from vit import VisionTransformer
from effnet import EfficientNet
from resnet import ResNet

model = EfficientNet()

model.load('data/EffNet.pth')

image = Image.open('image.jpg').convert('RGB')

prediction = model.predict(image)

print('Prediction', prediction)



