"""
NYA Anomaly & Unsupervised CLI Tool
Command-line interface for PatchCore anomaly detection and ResNet dataset clustering.
"""

import os
import sys
import argparse
import json
import cv2
from .patchcore import PatchCoreDetector
from .unsupervised_cluster import ResNetCluster


def main():
    parser = argparse.ArgumentParser(description="NYA ResNet 無監督二分類與異常檢測工具")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # 子命令 1: build-ok (構建良品特徵記憶庫)
    p_build = subparsers.add_parser("build-ok", help="利用良品圖片資料夾構建 PatchCore 記憶庫模型")
    p_build.add_argument("--ok-dir", required=True, help="良品 (OK) 圖片資料夾路徑")
    p_build.add_argument("--save-model", default="output/patchcore_model.pt", help="模型保存路徑")
    p_build.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet34", "resnet50", "wide_resnet50_2"])
    p_build.add_argument("--coreset", type=float, default=0.05, help="核心集壓縮率 (預設 0.05)")

    # 子命令 2: predict (推理待測圖片 / 資料夾)
    p_pred = subparsers.add_parser("predict", help="對待測圖片進行異常檢測並輸出熱力圖")
    p_pred.add_argument("--model", required=True, help="已訓練的 PatchCore 模型檔 (.pt)")
    p_pred.add_argument("--source", required=True, help="待測單張圖片或圖片資料夾路徑")
    p_pred.add_argument("--output-dir", default="output/anomaly_results", help="結果與熱力圖保存資料夾")
    p_pred.add_argument("--threshold", type=float, default=None, help="手動指定門檻值 (若不指定則採用模型建議值)")

    # 子命令 3: cluster (混合圖片無監督二分類分群)
    p_clust = subparsers.add_parser("cluster", help="對混合未標註圖片資料夾執行無監督二分類分群")
    p_clust.add_argument("--input-dir", required=True, help="待分群的圖片資料夾路徑")
    p_clust.add_argument("--output-dir", default="output/clustered_dataset", help="分群結果輸出資料夾")
    p_clust.add_argument("--k", type=int, default=2, help="聚類數量 (預設 2)")
    p_clust.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet34", "resnet50", "wide_resnet50_2"])

    args = parser.parse_args()

    if args.command == "build-ok":
        detector = PatchCoreDetector(backbone=args.backbone, coreset_ratio=args.coreset)
        stats = detector.fit(args.ok_dir)
        saved_path = detector.save_model(args.save_model)
        print(f"\n✨ 模型已成功保存至: {saved_path}")
        print(f"📊 良品基準統計: {json.dumps(stats, indent=2)}")

    elif args.command == "predict":
        detector = PatchCoreDetector()
        detector.load_model(args.model)
        os.makedirs(args.output_dir, exist_ok=True)

        if os.path.isdir(args.source):
            exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
            files = [os.path.join(args.source, f) for f in os.listdir(args.source) if f.lower().endswith(exts)]
        else:
            files = [args.source]

        print(f"\n🔍 開始檢測 {len(files)} 張圖片...")
        results = []
        for p in files:
            res = detector.predict(p, threshold=args.threshold, return_heatmap=True)
            fname = os.path.basename(p)
            out_img_path = os.path.join(args.output_dir, f"heat_{fname}")
            if "overlay_bgr" in res and res["overlay_bgr"] is not None:
                # Unicode-safe write
                is_success, buf = cv2.imencode(".jpg", res["overlay_bgr"])
                if is_success:
                    buf.tofile(out_img_path)

            print(f"  [{res['decision']}] {fname} | Score: {res['anomaly_score']} | Heatmap: {out_img_path}")
            res_summary = {
                "file": fname,
                "score": res["anomaly_score"],
                "decision": res["decision"],
                "is_anomaly": res["is_anomaly"]
            }
            results.append(res_summary)

        report_path = os.path.join(args.output_dir, "report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n📄 檢測報告已輸出至: {report_path}")

    elif args.command == "cluster":
        clusterer = ResNetCluster(backbone=args.backbone, n_clusters=args.k)
        res = clusterer.cluster_dataset(args.input_dir, output_folder=args.output_dir)
        print(f"\n✨ 分群完成！統計: {res['counts']}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
