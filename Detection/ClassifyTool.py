"""
=============================================================================
NYA AI Studio - 工業二分類即時監控與 HTTP 主站伺服器 (ClassifyTool)
功能:
  1. 固定建立並即時監聽《verify》目錄 (及其所有子資料夾)。
  2. 預設從《weight》目錄讀取二分類模型權重檔 (weight/best.pt)。
  3. 發現新影像立即執行二分類推論 (OK / NG)。
  4. 檢測完成後立即刪除該圖片檔案。
  5. 架設 HTTP 主站伺服器 (192.168.1.4 / 0.0.0.0)，供客戶端查詢或主動推送結果。
  6. 支援獨立單一類運行與打包為獨立 .exe 執行檔。
=============================================================================
"""

import os
import sys
import time
import json
import socket
import shutil
import threading
import urllib.request
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import cv2
import numpy as np
from ultralytics import YOLO

# ── 判斷執行環境與基礎目錄 ──────────────────────────────
if getattr(sys, 'frozen', False):
    # 若被打包為 .exe，以 exe 所在的目錄作為基礎路徑
    BASE_DIR = os.path.dirname(sys.executable)
else:
    # 源碼執行時，以專案根目錄為基礎路徑
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..")) if "Detection" in CURRENT_DIR else CURRENT_DIR

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ── 預設目錄與參數配置 ────────────────────────────────────
# 監聽目錄（固定為 base 目錄下的 《verify》）
VERIFY_DIR = os.path.join(BASE_DIR, "verify")

# 模型權重目錄（固定為 base 目錄下的 《weight》）
WEIGHT_DIR = os.path.join(BASE_DIR, "weight")

# HTTP 主站伺服器綁定 IP 與連接埠
SERVER_HOST = "0.0.0.0"       # 綁定 0.0.0.0 同時接受本機、192.168.1.4 及所有區域網網路介面
SERVER_PORT = 8080            # 主站 HTTP 端口

# 若需要主動向 Client 推送結果，可在此設定 Client 端接收網址 (如 http://192.168.1.4:8000/api/result)
CLIENT_PUSH_URL = "http://192.168.1.4:8000/result"


# ── 獲取本機所有網路 IP ──────────────────────────────────
def get_local_ip_list():
    ips = []
    try:
        hostname = socket.gethostname()
        for ip in socket.gethostbyname_ex(hostname)[2]:
            if not ip.startswith("127.") and ip not in ips:
                ips.append(ip)
    except Exception:
        pass
    if "192.168.1.4" not in ips:
        ips.insert(0, "192.168.1.4")
    return ips


# ── 全域狀態管理 ──────────────────────────────────────────
class ServerState:
    def __init__(self):
        self.lock = threading.Lock()
        self.total_count = 0
        self.ok_count = 0
        self.ng_count = 0
        self.latest_result = {
            "result": "NONE",
            "confidence": 0.0,
            "filename": "",
            "speed_ms": 0.0,
            "timestamp": ""
        }
        self.history = []

    def update(self, filename, label, conf, speed_ms):
        with self.lock:
            self.total_count += 1
            if label == "OK":
                self.ok_count += 1
            else:
                self.ng_count += 1

            now_str = time.strftime("%Y-%m-%d %H:%M:%S")
            item = {
                "result": label,
                "confidence": round(float(conf), 4),
                "filename": filename,
                "speed_ms": round(float(speed_ms), 2),
                "timestamp": now_str
            }
            self.latest_result = item
            self.history.append(item)
            if len(self.history) > 100:
                self.history.pop(0)
            return item

    def get_summary(self):
        with self.lock:
            yield_rate = (self.ok_count / self.total_count * 100.0) if self.total_count > 0 else 0.0
            return {
                "status": "online",
                "server_ips": get_local_ip_list(),
                "port": SERVER_PORT,
                "verify_dir": VERIFY_DIR,
                "weight_dir": WEIGHT_DIR,
                "total_checked": self.total_count,
                "ok_count": self.ok_count,
                "ng_count": self.ng_count,
                "yield_rate_pct": round(yield_rate, 2),
                "latest_result": self.latest_result
            }


GLOBAL_STATE = ServerState()
GLOBAL_CLASSIFIER = None


# ── 二分類器封裝 ──────────────────────────────────────────
class BinaryClassifier:
    def __init__(self, model_path=None, device=None):
        if model_path is None or not os.path.exists(model_path):
            model_path = self._find_model_path()
        
        self.model_path = model_path
        self.device = device
        print(f"📦 [AI Model] 載入二分類權重檔案: {self.model_path}", flush=True)
        self.model = YOLO(self.model_path)

    def _find_model_path(self):
        os.makedirs(WEIGHT_DIR, exist_ok=True)
        
        # 1. 優先檢查 《weight》 目錄下的 best.pt 或其它 .pt 模型
        weight_best = os.path.join(WEIGHT_DIR, "best.pt")
        if os.path.exists(weight_best):
            return weight_best

        for f in os.listdir(WEIGHT_DIR):
            if f.endswith(".pt"):
                return os.path.join(WEIGHT_DIR, f)

        # 2. 檢查 runs/classify/ 最新訓練模型並自動同步至 《weight》 目錄
        runs_classify = os.path.join(BASE_DIR, "runs", "classify")
        if os.path.exists(runs_classify):
            candidates = []
            for root, _, files in os.walk(runs_classify):
                if "best.pt" in files:
                    pt_p = os.path.join(root, "best.pt")
                    candidates.append((os.path.getmtime(pt_p), pt_p))
            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                latest_best = candidates[0][1]
                try:
                    shutil.copy2(latest_best, weight_best)
                    print(f"💾 [Auto-Sync] 已自動將最新訓練模型複製至: {weight_best}", flush=True)
                    return weight_best
                except Exception:
                    return latest_best

        # 3. 檢查 base_dir/weights/
        weights_dir = os.path.join(BASE_DIR, "weights")
        if os.path.exists(weights_dir):
            for f in os.listdir(weights_dir):
                if f.endswith(".pt") and ("cls" in f.lower() or "resnet" in f.lower() or "best" in f.lower()):
                    pt_p = os.path.join(weights_dir, f)
                    try:
                        shutil.copy2(pt_p, weight_best)
                        return weight_best
                    except Exception:
                        return pt_p

        return weight_best

    def predict_image(self, img_input, imgsz=512):
        kwargs = {"imgsz": imgsz, "verbose": False}
        if self.device is not None:
            kwargs["device"] = self.device
        
        t0 = time.time()
        results = self.model.predict(img_input, **kwargs)
        elapsed_ms = (time.time() - t0) * 1000.0

        probs = results[0].probs
        if probs is not None:
            top1_idx = int(probs.top1)
            raw_label = str(results[0].names.get(top1_idx, top1_idx)).upper()
            conf = float(probs.top1conf)
            label = "OK" if "OK" in raw_label else "NG"
        else:
            label = "OK"
            conf = 1.0

        return label, conf, elapsed_ms


# ── HTTP 請求處理器 ──────────────────────────────────────
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class MasterHTTPRequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, data, status_code=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")

        # 1. 取得主站完整狀態與良率統計
        if path in ("", "/status", "/api/status"):
            self._send_json(GLOBAL_STATE.get_summary())

        # 2. 取得最新一張檢測結果 (OK / NG)
        elif path in ("/result", "/latest", "/api/result", "/api/latest"):
            summary = GLOBAL_STATE.get_summary()
            self._send_json(summary["latest_result"])

        # 3. 取得歷史紀錄清單
        elif path in ("/history", "/api/history"):
            with GLOBAL_STATE.lock:
                self._send_json({"history": GLOBAL_STATE.history})

        else:
            self._send_json({
                "error": f"未知路徑: {self.path}",
                "help": [
                    "/result  (獲取最新 OK/NG 判定)",
                    "/status  (獲取良率與統計數據)",
                    "/history (獲取歷史檢測列表)",
                    "/predict (HTTP POST 圖片二進位檢測)"
                ]
            }, status_code=404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")

        # 支援 Client 直接透過 HTTP POST 圖片進行即時檢測
        if path in ("/predict", "/api/predict"):
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                if content_length <= 0:
                    self._send_json({"error": "請求體為空"}, status_code=400)
                    return

                post_data = self.rfile.read(content_length)
                nparr = np.frombuffer(post_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is None:
                    self._send_json({"error": "無法解碼圖片資料"}, status_code=400)
                    return

                if GLOBAL_CLASSIFIER is None:
                    self._send_json({"error": "模型尚未初始化完成"}, status_code=503)
                    return

                label, conf, speed_ms = GLOBAL_CLASSIFIER.predict_image(img)
                res_item = GLOBAL_STATE.update("http_post_image", label, conf, speed_ms)
                
                self._send_json({
                    "result": label,
                    "confidence": conf,
                    "speed_ms": speed_ms,
                    "timestamp": res_item["timestamp"]
                })
            except Exception as e:
                self._send_json({"error": str(e)}, status_code=500)
        else:
            self._send_json({"error": "不支援的 POST 路徑"}, status_code=404)

    def log_message(self, format, *args):
        pass


def create_http_server(host=SERVER_HOST, port=SERVER_PORT):
    server = ThreadedHTTPServer((host, port), MasterHTTPRequestHandler)
    return server


# ── 主動推送結果至 Client ─────────────────────────────────
def push_result_to_client(result_data, client_url=CLIENT_PUSH_URL):
    if not client_url:
        return
    try:
        req_data = json.dumps(result_data).encode("utf-8")
        req = urllib.request.Request(
            client_url,
            data=req_data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            pass
    except Exception:
        pass


# ── 《verify》目錄檔案即時監控與檢測流程 ──────────────────
def watch_verify_directory(verify_dir=VERIFY_DIR):
    os.makedirs(verify_dir, exist_ok=True)
    img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')

    while True:
        try:
            found_files = []
            if os.path.exists(verify_dir):
                for root, _, files in os.walk(verify_dir):
                    for f in files:
                        if f.lower().endswith(img_exts):
                            found_files.append(os.path.join(root, f))

            if not found_files:
                time.sleep(0.05)
                continue

            for file_path in found_files:
                if not os.path.exists(file_path):
                    continue

                # 確保檔案寫入完整 (檢查檔案大小是否穩定)
                file_size = -1
                for _ in range(5):
                    try:
                        cur_size = os.path.getsize(file_path)
                        if cur_size > 0 and cur_size == file_size:
                            break
                        file_size = cur_size
                        time.sleep(0.02)
                    except Exception:
                        time.sleep(0.02)

                # 讀取圖片
                img = None
                try:
                    data = np.fromfile(file_path, dtype=np.uint8)
                    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                except Exception as e:
                    print(f"⚠️ [讀取異常] 無法讀取 {file_path}: {e}", flush=True)

                if img is None:
                    try: os.remove(file_path)
                    except Exception: pass
                    continue

                # 執行二分類推論
                fname = os.path.basename(file_path)
                label, conf, speed_ms = GLOBAL_CLASSIFIER.predict_image(img)

                # 檢測完成，立即刪除該張圖像
                try:
                    os.remove(file_path)
                    del_status = "已刪除 🗑️"
                except Exception as e:
                    del_status = f"刪除失敗 ({e})"

                # 更新全域狀態
                res_item = GLOBAL_STATE.update(fname, label, conf, speed_ms)

                # 控制台輸出
                color_tag = "🟢 [OK]" if label == "OK" else "🔴 [NG]"
                now_t = time.strftime("%H:%M:%S")
                print(f"[{now_t}] 📸 檢測檔案: {fname} -> {color_tag} (信心度: {conf*100:.2f}%, 耗時: {speed_ms:.1f}ms) | {del_status}", flush=True)

                # 主動推送給 Client (若有設定)
                threading.Thread(target=push_result_to_client, args=(res_item,), daemon=True).start()

        except Exception as ex:
            print(f"❌ [監控迴圈異常]: {ex}", flush=True)
            time.sleep(0.5)


# ── 主程式入口 ───────────────────────────────────────────
def main():
    global GLOBAL_CLASSIFIER

    # 1. 確保 《verify》 與 《weight》 目錄存在
    os.makedirs(VERIFY_DIR, exist_ok=True)
    os.makedirs(WEIGHT_DIR, exist_ok=True)

    # 2. 獲取本機 IP 清單
    ip_list = get_local_ip_list()
    primary_ip = ip_list[0] if ip_list else "192.168.1.4"

    # 3. 初始化 AI 模型 (優先自 weight/ 目錄載入)
    GLOBAL_CLASSIFIER = BinaryClassifier()

    # 4. 印出完整資訊看板
    print("=" * 70, flush=True)
    print("      NYA AI Studio - 工業二分類 (OK / NG) 主站服務系統", flush=True)
    print("=" * 70, flush=True)
    print(f"📁 監控目錄路徑 (Verify Dir):", flush=True)
    print(f"   └── {VERIFY_DIR}", flush=True)
    print(f"📦 模型權重路徑 (Weight File):", flush=True)
    print(f"   └── {GLOBAL_CLASSIFIER.model_path}", flush=True)
    print(f"🌐 HTTP 主站服務網址 (Server URLs):", flush=True)
    for ip in ip_list:
        print(f"   └── http://{ip}:{SERVER_PORT}", flush=True)
    print(f"   └── http://localhost:{SERVER_PORT}", flush=True)
    print(f"📡 API 呼叫端點 (API Endpoints):", flush=True)
    print(f"   ├── 獲取最新結果 : GET  http://{primary_ip}:{SERVER_PORT}/result", flush=True)
    print(f"   ├── 獲取良率統計 : GET  http://{primary_ip}:{SERVER_PORT}/status", flush=True)
    print(f"   ├── 獲取歷史紀錄 : GET  http://{primary_ip}:{SERVER_PORT}/history", flush=True)
    print(f"   └── 圖片直接推論 : POST http://{primary_ip}:{SERVER_PORT}/predict", flush=True)
    print("=" * 70, flush=True)

    # 5. 啟動 HTTP 主站伺服器 (即刻綁定連接埠)
    httpd = create_http_server(SERVER_HOST, SERVER_PORT)
    http_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    http_thread.start()

    print(f"👀 [Directory Watcher] 即時監聽已啟動！", flush=True)
    print(f"   💡 將圖片放入 verify/ 目錄即可自動觸發檢測，檢測後自動刪除並更新 OK/NG 結果！\n", flush=True)

    # 6. 主線程執行 verify 目錄即時監控
    watch_verify_directory(VERIFY_DIR)


if __name__ == "__main__":
    main()
