"""
DeepVision — Inference & Grad-CAM Visualization
Author: Shebin S Illikkal | Shebinsillikkal@gmail.com
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from models.classifier import DeepVisionClassifier
import matplotlib.pyplot as plt
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, x, class_idx=None):
        self.model.eval()
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, x.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

class DeepVisionPredictor:
    def __init__(self, model_path: str, num_classes: int, class_names: list):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DeepVisionClassifier(num_classes=num_classes, pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()
        self.class_names = class_names
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        target_layer = list(self.model.backbone.layer4.children())[-1]
        self.gradcam = GradCAM(self.model, target_layer)

    def predict(self, image_path: str) -> dict:
        img = Image.open(image_path).convert('RGB')
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx = probs.argmax()
        return {
            'prediction': self.class_names[pred_idx],
            'confidence': float(probs[pred_idx]),
            'probabilities': {c: float(p) for c, p in zip(self.class_names, probs)}
        }

    def explain(self, image_path: str, save_path: str = None):
        img = Image.open(image_path).convert('RGB')
        x = self.transform(img).unsqueeze(0).to(self.device)
        x.requires_grad_(True)
        cam = self.gradcam.generate(x)
        img_np = np.array(img.resize((224, 224)))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_np, 0.6, heatmap[:, :, ::-1], 0.4, 0)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_np); axes[0].set_title("Original")
        axes[1].imshow(cam, cmap='jet'); axes[1].set_title("Grad-CAM")
        axes[2].imshow(overlay); axes[2].set_title("Overlay")
        for ax in axes: ax.axis('off')
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig
