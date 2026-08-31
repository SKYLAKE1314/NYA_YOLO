import os
import sys
import time
import json
import threading
import urllib.request
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import cv2
import numpy as np
from ultralytics import YOLO


class ClassifyTool:
    def __init__(self, host="0.0.0.0", port=8080, client_url=None, verify_dir=None, weight_dir=None):
        if getattr(sys, 'frozen', False):
            self.base_dir = os.path.dirname(sys.executable)
        else:
            cur = os.path.dirname(os.path.abspath(__file__))
            self.base_dir = os.path.abspath(os.path.join(cur, "..")) if "Detection" in cur else cur

        if self.base_dir not in sys.path:
            sys.path.insert(0, self.base_dir)

        self.verify_dir = verify_dir or os.path.join(self.base_dir, "verify")
        self.weight_dir = weight_dir or os.path.join(self.base_dir, "weight")
        self.host = host
        self.port = port
        self.client_url = client_url

        self.lock = threading.Lock()
        self.latest_result = {"results": [], "data": {}, "latest": "NONE", "id": 0}

        os.makedirs(self.verify_dir, exist_ok=True)
        os.makedirs(self.weight_dir, exist_ok=True)

        self.server = self._create_http_server()
        self.http_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.http_thread.start()

        self.model_path = self._find_model()
        print(f"載入權重檔案: {self.model_path}", flush=True)
        self.model = YOLO(self.model_path)

    def _find_model(self):
        best_p = os.path.join(self.weight_dir, "best.pt")
        if os.path.exists(best_p):
            return best_p
        for f in os.listdir(self.weight_dir):
            if f.endswith(".pt"):
                return os.path.join(self.weight_dir, f)
        return best_p

    def predict_image(self, img_input, imgsz=512):
        results = self.model.predict(img_input, imgsz=imgsz, verbose=False)
        probs = results[0].probs
        if probs is not None:
            top1_idx = int(probs.top1)
            raw = str(results[0].names.get(top1_idx, top1_idx)).upper()
            conf = float(probs.top1conf)
            label = "OK" if "OK" in raw else "NG"
        else:
            label = "OK"
            conf = 1.0
        return label, conf

    def push_to_client(self, payload):
        if not self.client_url:
            return
        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(self.client_url, data=req_data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=0.5) as resp:
                pass
        except Exception:
            pass

    def process_images(self):
        img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')
        found_files = []
        if os.path.exists(self.verify_dir):
            for root, _, files in os.walk(self.verify_dir):
                for f in files:
                    if f.lower().endswith(img_exts):
                        found_files.append(os.path.join(root, f))

        if not found_files:
            return

        batch_results = []
        batch_dict = {}

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
            except Exception:
                pass

            if img is None:
                try: os.remove(file_path)
                except Exception: pass
                continue

            fname = os.path.basename(file_path)
            stem, _ = os.path.splitext(fname)

            label, conf = self.predict_image(img)

            try:
                os.remove(file_path)
            except Exception:
                pass

            formatted_result = f"{stem}-{label}"
            batch_results.append(formatted_result)
            batch_dict[stem] = label

            now_t = time.strftime("%H:%M:%S")
            print(f"[{now_t}] {fname} -> [{formatted_result}] ({conf*100:.1f}%)", flush=True)

        if batch_results:
            payload = {
                "results": batch_results,
                "data": batch_dict,
                "latest": batch_results[-1],
                "id": int(time.time() * 1000)
            }
            with self.lock:
                self.latest_result = payload

            if self.client_url:
                threading.Thread(target=self.push_to_client, args=(payload,), daemon=True).start()

    def _create_http_server(self):
        tool_self = self

        class RequestHandler(BaseHTTPRequestHandler):
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
                    with tool_self.lock:
                        res = tool_self.latest_result
                    self._send_json(res)
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
                        label, conf = tool_self.predict_image(img)
                        res = {"result": f"image-{label}", "label": label, "confidence": conf}
                        self._send_json(res)
                    except Exception as e:
                        self._send_json({"error": str(e)}, status_code=500)
                else:
                    self._send_json({"error": "404 Not Found"}, status_code=404)

            def log_message(self, format, *args):
                pass

        class ThreadedServer(ThreadingMixIn, HTTPServer):
            daemon_threads = True

        return ThreadedServer((self.host, self.port), RequestHandler)

    def run(self):
        primary_ip = "192.168.1.4"
        print("=" * 60, flush=True)
        print("      NYA AI Studio - 二分類主站", flush=True)
        print("=" * 60, flush=True)
        print(f"監控目錄: {self.verify_dir}", flush=True)
        print(f"權重檔案: {self.model_path}", flush=True)
        print(f"服務網址: http://{primary_ip}:{self.port}/result", flush=True)
        print("=" * 60, flush=True)

        while True:
            try:
                self.process_images()
                time.sleep(0.02)
            except Exception as ex:
                print(f"異常: {ex}", flush=True)
                time.sleep(0.2)


if __name__ == "__main__":
    tool = ClassifyTool()
    tool.run()
