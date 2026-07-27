# Image Classification Telegram Bot

Telegram bot that classifies user-sent images in real time.
Trained and compared three architectures in PyTorch.

## Results
| Model              | Accuracy |
|--------------------|----------|
| ResNet-18          | 0.81     |
| EfficientNet       | 0.85     |
| Vision Transformer | **0.87** |

Best model: **Vision Transformer — 87% accuracy** (binary classification).

## Approach
- Collected and preprocessed a custom image dataset (~N images).
- Fine-tuned ResNet-18, EfficientNet and ViT.
- Integrated the best model into a Telegram bot for real-time inference.

## Tech stack
Python · PyTorch · TorchVision · python-telegram-bot
