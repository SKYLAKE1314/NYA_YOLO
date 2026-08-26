"""
PatchCore: Memory Bank Based Unsupervised Anomaly Detection & Heatmap Localization
Uses Pretrained ResNet Intermediate Layer Patch Representations + Coreset Subsampling.
"""

import os
import time
import math
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from .resnet_extractor import ResNetFeatureExtractor


class PatchCoreDetector:
    """
    PatchCore 無監督異常檢測器
    - 訓練階段：只需提供良品 (OK) 圖片，抽取局部 Patch 特徵並壓縮建立記憶庫
    - 推理階段：即時比對待測圖特徵距離，輸出 OK/NG 判定與像素級瑕疵熱力定位圖 (Heatmap)
    """

    def __init__(self, backbone="resnet18", return_layers=("layer2", "layer3"),
                 img_size=(256, 256), coreset_ratio=0.05, device=None):
        self.backbone_name = backbone
        self.return_layers = return_layers
        self.img_size = img_size
        self.coreset_ratio = coreset_ratio

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.extractor = ResNetFeatureExtractor(
            backbone_name=backbone,
            pretrained=True,
            return_layers=return_layers,
            device=self.device
        )

        self.memory_bank = None          # (N_coreset_patches, Feature_dim)
        self.threshold = 0.5             # 判定良品/不良品的預設門檻值
        self.normal_score_stats = {}     # 良品統計分數 (mean, max, std)
        self.spatial_shape = None        # (h, w) 特徵圖網格大小

    def _greedy_coreset_subsampling(self, features, target_size):
        """
        快速 GPU/CPU 貪婪核心集取樣 (Greedy Coreset Subsampling)
        壓縮記憶庫大小，保留 99% 的覆蓋度並大幅提升檢測速度
        features: (N, D)
        target_size: 取樣後保留的特徵數量
        """
        n_samples, feat_dim = features.shape
        if target_size >= n_samples:
            return features

        device = features.device
        # 降維以加速距離計算 (Johnson-Lindenstrauss Random Projection)
        proj_dim = min(128, feat_dim)
        if proj_dim < feat_dim:
            proj_matrix = torch.randn(feat_dim, proj_dim, device=device)
            proj_matrix = F.normalize(proj_matrix, p=2, dim=0)
            proj_features = torch.matmul(features, proj_matrix)
        else:
            proj_features = features

        # 貪婪取樣 (Minimax distance sampling)
        selected_indices = [0]
        # 初始化所有點到第 0 個點的距離
        dist_matrix = torch.norm(proj_features - proj_features[0], dim=1, keepdim=True)
        min_distances = dist_matrix

        for _ in range(1, target_size):
            # 挑選當前到所有已選點集合中最遠的點
            next_idx = torch.argmax(min_distances).item()
            selected_indices.append(next_idx)

            # 更新所有點到新選中點的距離
            new_dist = torch.norm(proj_features - proj_features[next_idx], dim=1, keepdim=True)
            min_distances = torch.min(min_distances, new_dist)

        selected_indices = torch.tensor(selected_indices, dtype=torch.long, device=device)
        return features[selected_indices]

    def fit(self, ok_image_inputs, batch_size=16, log_func=print):
        """
        構建良品記憶庫 (Training / Memory Bank Construction)
        ok_image_inputs: 圖片路徑列表、或包含 OK 圖片的資料夾路徑
        """
        start_time = time.time()

        if isinstance(ok_image_inputs, str) and os.path.isdir(ok_image_inputs):
            exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
            image_paths = [os.path.join(ok_image_inputs, f) for f in os.listdir(ok_image_inputs)
                           if f.lower().endswith(exts)]
        elif isinstance(ok_image_inputs, (list, tuple)):
            image_paths = ok_image_inputs
        else:
            raise ValueError("ok_image_inputs 必須為包含圖片的資料夾路徑或路徑列表")

        if not image_paths:
            raise ValueError("未找到任何有效良品圖片！")

        log_func(f"🚀 開始提取 {len(image_paths)} 張良品特徵以構建 PatchCore 記憶庫...")

        all_patch_features = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            tensors = []
            for p in batch_paths:
                t = self.extractor.preprocess_image(p, target_size=self.img_size)
                tensors.append(t)
            batch_tensor = torch.stack(tensors, dim=0)

            patch_feats, spatial_shape = self.extractor.extract_patch_features(batch_tensor)
            self.spatial_shape = spatial_shape
            b, n_patches, c = patch_feats.shape
            all_patch_features.append(patch_feats.reshape(b * n_patches, c))

        # 合併所有 Patch 特徵: (Total_Patches, Feature_Dim)
        merged_features = torch.cat(all_patch_features, dim=0)
        total_patches = merged_features.shape[0]

        # 核心集壓縮
        target_size = max(100, int(total_patches * self.coreset_ratio))
        log_func(f"⚡ 正在進行 Coreset 取樣壓縮 (原始 {total_patches} 點 -> 壓縮至 {target_size} 點)...")
        
        self.memory_bank = self._greedy_coreset_subsampling(merged_features, target_size)
        log_func(f"✅ 良品記憶庫構建完成！記憶庫大小: {self.memory_bank.shape}")

        # 計算良品基準分數以自動設定預設閾值
        log_func("🔍 計算良品自身基準分數分佈...")
        scores = []
        # 取樣前 30 張計算基準
        sample_paths = image_paths[:min(30, len(image_paths))]
        for p in sample_paths:
            res = self.predict(p, return_heatmap=False)
            scores.append(res["anomaly_score"])

        scores = np.array(scores)
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        max_score = float(np.max(scores))

        # 預設閾值為良品最大值 + 3 * 標準差 (3-Sigma 準則)
        self.threshold = round(max_score + 3.0 * std_score, 4)
        self.normal_score_stats = {
            "mean": mean_score,
            "std": std_score,
            "max": max_score,
            "recommended_threshold": self.threshold
        }

        elapsed = time.time() - start_time
        log_func(f"✨ 訓練完成 (耗時 {elapsed:.2f}s) | 建議異常判定門檻值 (Threshold): {self.threshold}")
        return self.normal_score_stats

    @torch.no_grad()
    def predict(self, img_input, threshold=None, return_heatmap=True):
        """
        推理單張圖片
        輸出:
          - anomaly_score: 異常分數 (越高代表瑕疵越明顯)
          - is_anomaly: 是否判定為異常 (True: NG, False: OK)
          - heatmap: 2D numpy 陣列 (0~1)
          - overlay_bgr: 疊加熱力圖的 BGR OpenCV 影像
        """
        if self.memory_bank is None:
            raise RuntimeError("尚未構建良品記憶庫，請先執行 fit() 或 load()！")

        active_thresh = threshold if threshold is not None else self.threshold

        # 1. 影像預處理與特徵抽取
        tensor = self.extractor.preprocess_image(img_input, target_size=self.img_size)
        patch_feats, spatial_shape = self.extractor.extract_patch_features(tensor)
        self.spatial_shape = spatial_shape
        # patch_feats: (1, H*W, C)
        patches = patch_feats.squeeze(0)  # (H*W, C)

        # 2. 計算每個 patch 到記憶庫中所有特徵的歐氏距離 (Chunked 以免顯存溢出)
        # patches: (N, C), memory_bank: (M, C)
        chunk_size = 512
        min_dists = []

        for i in range(0, patches.shape[0], chunk_size):
            chunk = patches[i:i + chunk_size]
            # (chunk_size, M)
            dists = torch.cdist(chunk, self.memory_bank, p=2)
            chunk_min, _ = torch.min(dists, dim=1)
            min_dists.append(chunk_min)

        patch_scores = torch.cat(min_dists, dim=0)  # (H*W,)
        h, w = self.spatial_shape

        # 3. 影像層級異常分數 (Top-K 最大值均值，提升抗噪能力)
        topk_scores, _ = torch.topk(patch_scores, k=min(5, patch_scores.numel()))
        anomaly_score = float(topk_scores.mean().cpu().item())
        is_anomaly = anomaly_score > active_thresh

        result = {
            "anomaly_score": round(anomaly_score, 4),
            "is_anomaly": is_anomaly,
            "decision": "NG (Defect)" if is_anomaly else "OK (Normal)",
            "threshold": active_thresh
        }

        if not return_heatmap:
            return result

        # 4. 生成像素級熱力圖 (Heatmap)
        score_map = patch_scores.reshape(1, 1, h, w)
        # 上採樣至目標影像大小
        upsampled_map = F.interpolate(score_map, size=self.img_size, mode="bilinear", align_corners=False)
        heatmap_np = upsampled_map.squeeze().cpu().numpy()

        # 高斯平滑
        heatmap_np = cv2.GaussianBlur(heatmap_np, (15, 15), 4)

        # 正規化到 0~1 (以 threshold 作為基準刻度)
        norm_map = np.clip((heatmap_np - (active_thresh * 0.5)) / (active_thresh * 1.5 - active_thresh * 0.5 + 1e-6), 0.0, 1.0)
        result["heatmap"] = norm_map

        # 5. 生成視覺化疊加圖 (Overlay Image)
        if isinstance(img_input, str):
            orig_data = np.fromfile(img_input, dtype=np.uint8)
            orig_bgr = cv2.imdecode(orig_data, cv2.IMREAD_COLOR)
        elif isinstance(img_input, np.ndarray):
            orig_bgr = img_input.copy()
        elif isinstance(img_input, Image.Image):
            orig_bgr = cv2.cvtColor(np.array(img_input), cv2.COLOR_RGB2BGR)
        else:
            orig_bgr = None

        if orig_bgr is not None:
            orig_resized = cv2.resize(orig_bgr, self.img_size)
            heatmap_uint8 = np.uint8(255 * norm_map)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(orig_resized, 0.6, heatmap_color, 0.4, 0)

            # 標註判定文字
            label_text = f"{result['decision']} Score: {anomaly_score:.3f}"
            text_color = (0, 0, 255) if is_anomaly else (0, 255, 0)
            cv2.putText(overlay, label_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            result["overlay_bgr"] = overlay

        return result

    def save_model(self, save_path):
        """保存訓練好的記憶庫模型"""
        if self.memory_bank is None:
            raise RuntimeError("尚未訓練模型，無法保存！")

        state = {
            "backbone_name": self.backbone_name,
            "return_layers": self.return_layers,
            "img_size": self.img_size,
            "coreset_ratio": self.coreset_ratio,
            "memory_bank": self.memory_bank.cpu(),
            "threshold": self.threshold,
            "normal_score_stats": self.normal_score_stats,
            "spatial_shape": self.spatial_shape
        }
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        torch.save(state, save_path)
        return save_path

    def load_model(self, model_path):
        """載入已儲存的記憶庫模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型檔案: {model_path}")

        state = torch.load(model_path, map_location=self.device)
        self.backbone_name = state.get("backbone_name", "resnet18")
        self.return_layers = state.get("return_layers", ("layer2", "layer3"))
        self.img_size = state.get("img_size", (256, 256))
        self.coreset_ratio = state.get("coreset_ratio", 0.05)
        self.memory_bank = state["memory_bank"].to(self.device)
        self.threshold = state.get("threshold", 0.5)
        self.normal_score_stats = state.get("normal_score_stats", {})
        self.spatial_shape = state.get("spatial_shape", None)

        # 重新初始化特徵提取器
        self.extractor = ResNetFeatureExtractor(
            backbone_name=self.backbone_name,
            pretrained=True,
            return_layers=self.return_layers,
            device=self.device
        )
        return self
