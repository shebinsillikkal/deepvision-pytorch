"""
DeepVision — ViT + ResNet Ensemble Classifier
Author: Shebin S Illikkal | Shebinsillikkal@gmail.com
"""
import torch
import torch.nn as nn
import timm
from typing import Tuple

CLASSES = ['Normal','Pneumonia','Tuberculosis','Pleural Effusion','Cardiomegaly',
           'Atelectasis','Consolidation','Edema','Pneumothorax','Fracture',
           'Mass/Nodule','Fibrosis']

class DeepVisionClassifier(nn.Module):
    def __init__(self, num_classes: int = 12, pretrained: bool = True):
        super().__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained,
                                      num_classes=0, global_pool='token')
        vit_features = self.vit.num_features  # 768
        self.resnet = timm.create_model('resnet50', pretrained=pretrained,
                                         num_classes=0, global_pool='avg')
        resnet_features = self.resnet.num_features  # 2048
        self.fusion = nn.Sequential(
            nn.Linear(vit_features + resnet_features, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vit_out    = self.vit(x)
        resnet_out = self.resnet(x)
        combined   = torch.cat([vit_out, resnet_out], dim=1)
        return self.fusion(combined)

    def predict_with_confidence(self, x: torch.Tensor) -> Tuple[str, float]:
        self.eval()
        with torch.no_grad():
            logits = self(x)
            probs  = torch.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)
        return CLASSES[idx.item()], conf.item()

def get_model(checkpoint_path: str = None) -> DeepVisionClassifier:
    model = DeepVisionClassifier()
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state['model_state_dict'])
    return model
