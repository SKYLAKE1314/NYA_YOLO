"""
ResNet Unsupervised Binary / Multi Clustering & Dimensionality Reduction
Pure PyTorch implementation of K-Means and PCA for GPU-accelerated dataset sorting.
"""

import os
import shutil
import time
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from .resnet_extractor import ResNetFeatureExtractor


class PyTorchKMeans:
    """純 PyTorch GPU/CPU 實作的 K-Means++ 聚類演算法"""

    def __init__(self, n_clusters=2, max_iter=300, tol=1e-4, device=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.centroids = None

    def _kmeans_plus_plus_init(self, X):
        """K-Means++ 初始聚類中心選擇"""
        n_samples = X.shape[0]
        # 隨機挑選第一個中心
        first_idx = torch.randint(0, n_samples, (1,), device=self.device).item()
        centroids = [X[first_idx]]

        for _ in range(1, self.n_clusters):
            # 計算所有點到已選中心的最短距離
            centers = torch.stack(centroids, dim=0)  # (k, D)
            dists = torch.cdist(X, centers, p=2)    # (N, k)
            min_dist, _ = torch.min(dists, dim=1)   # (N,)
            probs = min_dist ** 2
            probs = probs / torch.sum(probs)

            # 依機率取樣下一個中心
            next_idx = torch.multinomial(probs, 1).item()
            centroids.append(X[next_idx])

        return torch.stack(centroids, dim=0)

    def fit_predict(self, X):
        """
        X: (N, D) 特徵張量
        回傳: labels (N,), centroids (K, D), distances (N, K)
        """
        X = X.to(self.device)
        self.centroids = self._kmeans_plus_plus_init(X)

        labels = torch.zeros(X.shape[0], dtype=torch.long, device=self.device)

        for _ in range(self.max_iter):
            # 1. 指派樣本到最近的聚類中心
            dists = torch.cdist(X, self.centroids, p=2)  # (N, K)
            new_labels = torch.argmin(dists, dim=1)

            # 2. 更新聚類中心
            new_centroids = []
            for k in range(self.n_clusters):
                mask = (new_labels == k)
                if torch.sum(mask) > 0:
                    new_centroids.append(X[mask].mean(dim=0))
                else:
                    new_centroids.append(self.centroids[k])
            new_centroids = torch.stack(new_centroids, dim=0)

            # 檢查收斂
            shift = torch.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            labels = new_labels

            if shift < self.tol:
                break

        final_dists = torch.cdist(X, self.centroids, p=2)
        return labels.cpu().numpy(), self.centroids.cpu().numpy(), final_dists.cpu().numpy()


class PyTorchPCA:
    """純 PyTorch SVD 實作的主成分分析 (PCA) 降維"""

    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit_transform(self, X):
        """
        X: (N, D) numpy 陣列或 torch 張量
        回傳: (N, n_components) 降維後坐標
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()

        # 中心化
        self.mean = torch.mean(X, dim=0, keepdim=True)
        X_centered = X - self.mean

        # SVD奇異值分解
        U, S, V = torch.pca_lowrank(X_centered, q=self.n_components, center=False)
        self.components = V[:, :self.n_components]
        transformed = torch.matmul(X_centered, self.components)
        return transformed.cpu().numpy()


class ResNetCluster:
    """
    ResNet 無監督資料集二分類與自動分群系統
    """

    def __init__(self, backbone="resnet18", n_clusters=2, img_size=(256, 256), device=None):
        self.backbone = backbone
        self.n_clusters = n_clusters
        self.img_size = img_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.extractor = ResNetFeatureExtractor(
            backbone_name=backbone,
            pretrained=True,
            device=self.device
        )

    def cluster_dataset(self, input_folder, output_folder=None, copy_files=True, log_func=print):
        """
        對無標註資料夾執行特徵抽取與無監督聚類
        """
        start_time = time.time()
        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

        if not os.path.exists(input_folder):
            raise FileNotFoundError(f"找不到輸入資料夾: {input_folder}")

        image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(exts)]
        if len(image_files) < self.n_clusters:
            raise ValueError(f"圖片數量 ({len(image_files)}) 少於聚類數 ({self.n_clusters})！")

        log_func(f"🚀 開始為 {len(image_files)} 張影像提取 ResNet 全域特徵向量...")

        features_list = []
        valid_image_paths = []

        batch_size = 32
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i + batch_size]
            tensors = []
            for f in batch_files:
                p = os.path.join(input_folder, f)
                try:
                    t = self.extractor.preprocess_image(p, target_size=self.img_size)
                    tensors.append(t)
                    valid_image_paths.append(p)
                except Exception as e:
                    log_func(f"⚠️ 讀取影像失敗 {f}: {e}")

            if tensors:
                batch_tensor = torch.stack(tensors, dim=0)
                feats = self.extractor.extract_global_embedding(batch_tensor)
                features_list.append(feats)

        all_features = torch.cat(features_list, dim=0)
        log_func(f"✨ 特徵提取完成！特徵維度: {all_features.shape}")

        # 執行 K-Means 聚類
        log_func(f"🔍 執行 K-Means (K={self.n_clusters}) 聚類...")
        kmeans = PyTorchKMeans(n_clusters=self.n_clusters, device=self.device)
        labels, centroids, distances = kmeans.fit_predict(all_features)

        # 執行 2D PCA 降維 (用於視覺化)
        log_func("📊 執行 PCA 2D 降維投影...")
        pca = PyTorchPCA(n_components=2)
        coords_2d = pca.fit_transform(all_features)

        # 統計各類數量
        counts = {}
        for k in range(self.n_clusters):
            counts[f"cluster_{k}"] = int(np.sum(labels == k))

        # 自動劃分與複製到輸出資料夾
        if output_folder:
            log_func(f"📂 正在將分類結果輸出至: {output_folder}")
            os.makedirs(output_folder, exist_ok=True)
            for k in range(self.n_clusters):
                os.makedirs(os.path.join(output_folder, f"cluster_{k}"), exist_ok=True)

            for p, label in zip(valid_image_paths, labels):
                fname = os.path.basename(p)
                dst = os.path.join(output_folder, f"cluster_{label}", fname)
                if copy_files:
                    shutil.copy2(p, dst)

        elapsed = time.time() - start_time
        log_func(f"✅ 無監督二分類聚類完成！耗時: {elapsed:.2f}s | 分群結果: {counts}")

        return {
            "counts": counts,
            "labels": labels.tolist(),
            "image_paths": valid_image_paths,
            "pca_coords_2d": coords_2d.tolist(),
            "output_folder": output_folder,
            "elapsed": elapsed
        }
