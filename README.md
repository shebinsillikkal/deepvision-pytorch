# DeepVision — Medical Image Classifier

> Vision Transformer + ResNet ensemble for medical X-ray classification — **97.2% accuracy** on 12 disease categories. Used in clinical decision support.

## The Problem
Trained a vision model on X-ray scans to help flag potential issues for doctors to review. The Grad-CAM heatmaps showing what the model was "looking at" were what convinced the doctors to actually trust it.

## Results
- 97.2% classification accuracy across 12 categories
- Grad-CAM heatmaps for model explainability
- Trained on 120K labelled X-rays
- ONNX export for edge deployment

## Stack
```
PyTorch | torchvision | Grad-CAM | ONNX | Albumentations | timm
```

**Built by Shebin S Illikkal** — Shebinsillikkal@gmail.com
