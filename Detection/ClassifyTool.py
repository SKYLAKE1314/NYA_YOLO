import os
import sys
import time
import json
import socket
import threading
import urllib.request
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import cv2
import numpy as np
from ultralytics import YOLO

if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..")) if "Detection" in CURRENT_DIR else CURRENT_DIR

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

VERIFY_DIR = os.path.join(BASE_DIR, "verify")
WEIGHT_DIR = os.path.join(BASE_DIR, "weight")
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8080
CLIENT_PUSH_URL = "http://192.168.1.4:8000/result"


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


class ServerState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_result = {"result": "NONE", "confidence": 0.0, "speed_ms": 0.0}

    def update(self, label, conf, speed_ms):
        with self.lock:
            item = {
                "result": label,
                "confidence": round(float(conf), 4),
                "speed_ms": round(float(speed_ms), 2)
            }
            self.latest_result = item
            return item

    def get_latest(self):
        with self.lock:
            return self.latest_result


GLOBAL_STATE = ServerState()
GLOBAL_CLASSIFIER = None


class BinaryClassifier:
    def __init__(self, model_path=None, device=None):
        if model_path is None or not os.path.exists(model_path):
            model_path = self._find_model_path()
        
        self.model_path = model_path
        self.device = device
        print(f"載入權重檔案: {self.model_path}", flush=True)
        self.model = YOLO(self.model_path)

    def _find_model_path(self):
        os.makedirs(WEIGHT_DIR, exist_ok=True)
        weight_best = os.path.join(WEIGHT_DIR, "best.pt")
        if os.path.exists(weight_best):
            return weight_best

        for f in os.listdir(WEIGHT_DIR):
            if f.endswith(".pt"):
                return os.path.join(WEIGHT_DIR, f)
        
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


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class MasterHTTPRequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, data, status_code=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path in ("", "/result", "/latest", "/status"):
            self._send_json(GLOBAL_STATE.get_latest())
        else:
            self._send_json({"error": "404 Not Found"}, status_code=404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path in ("/predict", "/api/predict"):
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                if content_length <= 0:
                    self._send_json({"error": "empty body"}, status_code=400)
                    return

                post_data = self.rfile.read(content_length)
                nparr = np.frombuffer(post_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is None:
                    self._send_json({"error": "decode failed"}, status_code=400)
                    return

                if GLOBAL_CLASSIFIER is None:
                    self._send_json({"error": "model not ready"}, status_code=503)
                    return

                label, conf, speed_ms = GLOBAL_CLASSIFIER.predict_image(img)
                res_item = GLOBAL_STATE.update(label, conf, speed_ms)
                self._send_json(res_item)
            except Exception as e:
                self._send_json({"error": str(e)}, status_code=500)
        else:
            self._send_json({"error": "404 Not Found"}, status_code=404)

    def log_message(self, format, *args):
        pass


def create_http_server(host=SERVER_HOST, port=SERVER_PORT):
    return ThreadedHTTPServer((host, port), MasterHTTPRequestHandler)


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
        with urllib.request.urlopen(req, timeout=0.5) as resp:
            pass
    except Exception:
        pass


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
                time.sleep(0.02)
                continue

            for file_path in found_files:
                if not os.path.exists(file_path):
                    continue

                file_size = -1
                for _ in range(5):
                    try:
                        cur_size = os.path.getsize(file_path)
                        if cur_size > 0 and cur_size == file_size:
                            break
                        file_size = cur_size
                        time.sleep(0.01)
                    except Exception:
                        time.sleep(0.01)

                img = None
                try:
                    data = np.fromfile(file_path, dtype=np.uint8)
                    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                except Exception as e:
                    print(f"無法讀取 {file_path}: {e}", flush=True)

                if img is None:
                    try: os.remove(file_path)
                    except Exception: pass
                    continue

                fname = os.path.basename(file_path)
                label, conf, speed_ms = GLOBAL_CLASSIFIER.predict_image(img)

                try:
                    os.remove(file_path)
                except Exception:
                    pass

                res_item = GLOBAL_STATE.update(label, conf, speed_ms)
                now_t = time.strftime("%H:%M:%S")
                print(f"[{now_t}] {fname} -> [{label}] ({conf*100:.1f}%, {speed_ms:.1f}ms)", flush=True)

                threading.Thread(target=push_result_to_client, args=(res_item,), daemon=True).start()

        except Exception as ex:
            print(f"異常: {ex}", flush=True)
            time.sleep(0.2)


def main():
    global GLOBAL_CLASSIFIER

    os.makedirs(VERIFY_DIR, exist_ok=True)
    os.makedirs(WEIGHT_DIR, exist_ok=True)

    ip_list = get_local_ip_list()
    primary_ip = ip_list[0] if ip_list else "192.168.1.4"

    GLOBAL_CLASSIFIER = BinaryClassifier()

    print("=" * 60, flush=True)
    print("      NYA AI Studio - 二分類主站", flush=True)
    print("=" * 60, flush=True)
    print(f"監控目錄: {VERIFY_DIR}", flush=True)
    print(f"權重檔案: {GLOBAL_CLASSIFIER.model_path}", flush=True)
    print(f"服務網址: http://{primary_ip}:{SERVER_PORT}/result", flush=True)
    print("=" * 60, flush=True)

    httpd = create_http_server(SERVER_HOST, SERVER_PORT)
    http_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    http_thread.start()

    watch_verify_directory(VERIFY_DIR)


if __name__ == "__main__":
    main()
