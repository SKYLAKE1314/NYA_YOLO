"""
ResNet Feature Extractor for Unsupervised Anomaly Detection & Clustering
Supports multi-scale intermediate layer feature extraction and global GAP embeddings.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2


class ResNetFeatureExtractor(nn.Module):
    """
    通用 ResNet 特徵提取器
    支援提取中間層局部特徵圖 (用於 PatchCore / 異常檢測)
    與全域池化特徵向量 (用於無監督聚類)
    """

    def __init__(self, backbone_name="resnet18", pretrained=True, return_layers=("layer2", "layer3"), device=None):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.return_layers = list(return_layers)
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 載入 ResNet 模型架構
        if self.backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.model = models.resnet18(weights=weights)
        elif self.backbone_name == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            self.model = models.resnet34(weights=weights)
        elif self.backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.model = models.resnet50(weights=weights)
        elif self.backbone_name in ("wide_resnet50_2", "wideresnet50"):
            weights = models.Wide_ResNet50_2_Weights.DEFAULT if pretrained else None
            self.model = models.wide_resnet50_2(weights=weights)
        else:
            raise ValueError(f"不支援的 Backbone: {backbone_name}，請選擇 resnet18 / resnet34 / resnet50 / wide_resnet50_2")

        # 凍結所有參數 (無監督特徵提取無需計算梯度)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.to(self.device)

        # 註冊 Hook 以獲取中間層特徵
        self.features = {}
        self.hooks = []
        self._register_hooks()

        # 標準 ImageNet 預處理 Transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _register_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

        def hook_fn(layer_name):
            def _hook(module, input, output):
                self.features[layer_name] = output
            return _hook

        for name, module in self.model.named_children():
            if name in self.return_layers:
                self.hooks.append(module.register_forward_hook(hook_fn(name)))

    def preprocess_image(self, img_input, target_size=(256, 256)):
        """
        支援讀取檔案路徑、PIL Image、OpenCV BGR numpy 陣列
        """
        if isinstance(img_input, str):
            # Unicode-safe image reading for Windows
            try:
                img_data = np.fromfile(img_input, dtype=np.uint8)
                img_bgr = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    raise ValueError(f"無法解碼圖像: {img_input}")
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
            except Exception as e:
                pil_img = Image.open(img_input).convert("RGB")
        elif isinstance(img_input, np.ndarray):
            if len(img_input.shape) == 2:
                img_input = cv2.cvtColor(img_input, cv2.COLOR_GRAY2RGB)
            elif img_input.shape[2] == 3:
                # 假設 OpenCV 傳入為 BGR
                img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_input)
        elif isinstance(img_input, Image.Image):
            pil_img = img_input.convert("RGB")
        else:
            raise TypeError("img_input 必須為路徑字串、PIL Image 或 numpy ndarray")

        if target_size is not None:
            pil_img = pil_img.resize(target_size, Image.BILINEAR)

        tensor = self.transform(pil_img)
        return tensor

    @torch.no_grad()
    def extract_patch_features(self, tensor_batch):
        """
        提取多尺度 Patch 特徵 (用於 PatchCore 異常檢測)
        輸入: (B, 3, H, W)
        輸出: (B, N_patches, Feature_dim) 特徵張量
        """
        if tensor_batch.ndim == 3:
            tensor_batch = tensor_batch.unsqueeze(0)
        tensor_batch = tensor_batch.to(self.device)

        self.features.clear()
        _ = self.model(tensor_batch)

        extracted_maps = [self.features[layer] for layer in self.return_layers]

        # 統一各層特徵圖的空間解析度 (以第一層為基準進行自適應平均池化或插值)
        target_h, target_w = extracted_maps[0].shape[-2:]
        aligned_features = []

        for fmap in extracted_maps:
            if fmap.shape[-2:] != (target_h, target_w):
                # 局部平均池化以聚合局部鄰域資訊
                fmap_aligned = F.adaptive_avg_pool2d(fmap, (target_h, target_w))
            else:
                fmap_aligned = fmap
            aligned_features.append(fmap_aligned)

        # 拼接多層特徵: (B, C1+C2, H, W)
        concat_map = torch.cat(aligned_features, dim=1)
        b, c, h, w = concat_map.shape

        # 展平成 Patch 特徵向量: (B, H*W, C)
        patch_features = concat_map.permute(0, 2, 3, 1).reshape(b, h * w, c)
        return patch_features, (h, w)

    @torch.no_grad()
    def extract_global_embedding(self, tensor_batch):
        """
        提取全域 GAP 特徵向量 (用於無監督聚類)
        輸入: (B, 3, H, W)
        輸出: (B, Feature_dim) 正規化後的特徵向量
        """
        if tensor_batch.ndim == 3:
            tensor_batch = tensor_batch.unsqueeze(0)
        tensor_batch = tensor_batch.to(self.device)

        # 前向傳播至 avgpool
        x = self.model.conv1(tensor_batch)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        features = torch.flatten(x, 1)

        # L2 正規化
        norm_features = F.normalize(features, p=2, dim=1)
        return norm_features
