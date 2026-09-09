import os
import sys
import time
import json
import threading
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import cv2
import numpy as np
import torch
from ultralytics import YOLO


class ClassifyTool:
    def __init__(self, host="0.0.0.0", port=8080, client_url=None, verify_dir=None, weight_dir=None, max_workers=8, infer_batch_size=16):
        if getattr(sys, 'frozen', False):
            self.base_dir = os.path.dirname(sys.executable)
        else:
            cur = os.path.dirname(os.path.abspath(__file__))
            self.base_dir = os.path.abspath(os.path.join(cur, "..")) if "Detection" in cur else cur

        if self.base_dir not in sys.path:
            sys.path.insert(0, self.base_dir)

        self.verify_dir = verify_dir or os.path.join(self.base_dir, "verify")
        self.weight_dir = weight_dir or os.path.join(self.base_dir, "weights")
        self.host = host
        self.port = port
        self.client_url = client_url
        self.max_workers = max_workers
        self.infer_batch_size = infer_batch_size  # 防止 CUDA OOM 的分批推論上限

        self.state_lock = threading.Lock()
        self.latest_result = {"results": [], "data": {}, "latest": "NONE", "id": 0}

        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        os.makedirs(self.verify_dir, exist_ok=True)
        os.makedirs(self.weight_dir, exist_ok=True)

        self.server = self._create_http_server()
        self.http_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.http_thread.start()

        self.model_path = self._find_model()
        print(f"權重檔案: {self.model_path}", flush=True)

        self.device, self.model = self._init_backend(self.model_path)
        self.is_cuda = (self.device == "0")

        # 預熱 (Warm-up)
        print("正在預寫入記憶體", flush=True)
        try:
            dummy = np.zeros((512, 512, 3), dtype=np.uint8)
            self.model.predict(dummy, imgsz=512, device=self.device, verbose=False)
            print("寫入完成，準備好", flush=True)
        except Exception as e:
            print(f"寫入失敗: {e}", flush=True)

    def _find_model(self):
        best_p = os.path.join(self.weight_dir, "best.pt")
        if os.path.exists(best_p):
            return best_p
        for f in os.listdir(self.weight_dir):
            if f.endswith(".pt"):
                return os.path.join(self.weight_dir, f)
        return best_p

    def _init_backend(self, model_path):
        if torch.cuda.is_available():
            try:
                print(f"使用 CUDA: {torch.cuda.get_device_name(0)}", flush=True)
                model = YOLO(model_path)
                return "0", model
            except Exception as e:
                print(f"CUDA 調用失敗 ({e})，嘗試下一順位...", flush=True)

        try:
            import openvino
            stem, _ = os.path.splitext(model_path)
            ov_path = f"{stem}_openvino_model"
            if not os.path.exists(ov_path):
                print("轉換模型為 OpenVINO 格式...", flush=True)
                tmp_model = YOLO(model_path)
                tmp_model.export(format="openvino", imgsz=512, half=False)
            model = YOLO(ov_path, task="classify")
            print(f"啟用 OpenVINO 模型: {ov_path}", flush=True)
            return "cpu", model
        except Exception:
            pass

        print("使用 CPU 推論", flush=True)
        return "cpu", YOLO(model_path)

    def _read_and_clean_image(self, file_path):
        if not os.path.exists(file_path):
            return None

        # 防抖
        file_size = -1
        for _ in range(3):
            try:
                cur_size = os.path.getsize(file_path)
                if cur_size > 0 and cur_size == file_size:
                    break
                file_size = cur_size
                time.sleep(0.005)
            except Exception:
                time.sleep(0.005)

        img = None
        try:
            data = np.fromfile(file_path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            pass

        try:
            os.remove(file_path)
        except Exception:
            pass

        if img is None:
            return None

        fname = os.path.basename(file_path)
        stem, _ = os.path.splitext(fname)
        return {"stem": stem, "fname": fname, "img": img}

    def process_images(self):
        img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')
        found_files = []
        if os.path.exists(self.verify_dir):
            try:
                with os.scandir(self.verify_dir) as entries:
                    for entry in entries:
                        if entry.is_file() and entry.name.lower().endswith(img_exts):
                            found_files.append(entry.path)
            except Exception:
                pass

        if not found_files:
            return

        t_start = time.time()

        valid_items = list(filter(None, self.executor.map(self._read_and_clean_image, found_files)))
        if not valid_items:
            return

        t_read_done = time.time()

        results = []
        img_list = [item["img"] for item in valid_items]
        
        for i in range(0, len(img_list), self.infer_batch_size):
            chunk = img_list[i:i + self.infer_batch_size]
            chunk_res = self.model.predict(
                chunk,
                imgsz=512,
                device=self.device,
                verbose=False
            )
            results.extend(chunk_res)

        t_infer_done = time.time()

        # 3. 解析結果
        batch_results = []
        batch_dict = {}
        for item, res in zip(valid_items, results):
            stem = item["stem"]
            probs = res.probs
            if probs is not None:
                top1_idx = int(probs.top1)
                raw = str(res.names.get(top1_idx, top1_idx)).upper()
                conf = float(probs.top1conf)
                label = "OK" if "OK" in raw else "NG"
            else:
                label = "OK"
                conf = 1.0

            formatted = f"{stem}-{label}"
            batch_results.append(formatted)
            batch_dict[stem] = label

        total_ms = (t_infer_done - t_start) * 1000
        infer_ms = (t_infer_done - t_read_done) * 1000
        avg_ms = infer_ms / len(valid_items)
        now_t = time.strftime("%H:%M:%S")

        print(
            f"[{now_t}] 批次處理 {len(valid_items)} 張圖片 -> 總耗時: {total_ms:.1f}ms (推論: {infer_ms:.1f}ms, 平均: {avg_ms:.1f}ms/張)",
            flush=True
        )

        # 4. 更新狀態與推播
        payload = {
            "results": batch_results,
            "data": batch_dict,
            "latest": batch_results[-1],
            "id": int(time.time() * 1000)
        }
        with self.state_lock:
            self.latest_result = payload

        if self.client_url:
            self.executor.submit(self.push_to_client, payload)

        # 5. 清理 GPU 碎片快取
        if self.is_cuda and len(valid_items) > 32:
            torch.cuda.empty_cache()

    def push_to_client(self, payload):
        if not self.client_url:
            return
        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(self.client_url, data=req_data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=0.5):
                pass
        except Exception:
            pass

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
                path = urllib.parse.urlparse(self.path).path.rstrip("/")
                if path in ("", "/result", "/latest", "/status"):
                    with tool_self.state_lock:
                        res = tool_self.latest_result
                    self._send_json(res)
                else:
                    self._send_json({"error": "404 Not Found"}, status_code=404)

            def do_POST(self):
                path = urllib.parse.urlparse(self.path).path.rstrip("/")
                if path in ("/predict", "/api/predict"):
                    try:
                        content_length = int(self.headers.get("Content-Length", 0))
                        post_data = self.rfile.read(content_length)
                        img = cv2.imdecode(np.frombuffer(post_data, np.uint8), cv2.IMREAD_COLOR)
                        if img is None:
                            self._send_json({"error": "decode failed"}, status_code=400)
                            return

                        res = tool_self.model.predict(img, imgsz=512, device=tool_self.device, verbose=False)[0]
                        top1_idx = int(res.probs.top1) if res.probs else 0
                        raw = str(res.names.get(top1_idx, top1_idx)).upper()
                        label = "OK" if "OK" in raw else "NG"
                        conf = float(res.probs.top1conf) if res.probs else 1.0

                        self._send_json({"result": f"image-{label}", "label": label, "confidence": conf})
                    except Exception as e:
                        self._send_json({"error": str(e)}, status_code=500)
                else:
                    self._send_json({"error": "404 Not Found"}, status_code=404)

            def log_message(self, format, *args):
                pass

        class ThreadedServer(ThreadingMixIn, HTTPServer):
            daemon_threads = True

        try:
            return ThreadedServer((self.host, self.port), RequestHandler)
        except OSError as e:
            if getattr(e, 'winerror', None) == 10013 or getattr(e, 'errno', None) == 10013:
                for alt_p in [8088, 8000]:
                    try:
                        srv = ThreadedServer((self.host, alt_p), RequestHandler)
                        print(f"端口 {self.port} 被系統佔用或保留，改用端口: {alt_p}", flush=True)
                        self.port = alt_p
                        return srv
                    except Exception:
                        pass
            raise

    def run(self):
        print("=" * 60, flush=True)
        print("      二分類主站", flush=True)
        print("=" * 60, flush=True)
        print(f"硬件加速: {self.device if self.device != '0' else 'CUDA (Auto FP16)'}", flush=True)
        print(f"處理目錄: {self.verify_dir}", flush=True)
        print(f"拆分批次: {self.infer_batch_size} 張/批", flush=True)
        print("=" * 60, flush=True)

        while True:
            try:
                self.process_images()
                time.sleep(0.01)
            except Exception as ex:
                print(f"錯誤: {ex}", flush=True)
                time.sleep(0.1)


if __name__ == "__main__":
    tool = ClassifyTool(max_workers=8, infer_batch_size=32)
    tool.run()