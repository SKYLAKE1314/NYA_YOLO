import sys
import time
import json
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

HOST = "0.0.0.0"
PORT = 8000
MASTER_URL = "http://192.168.1.4:8080/result"


class ReceiverHandler(BaseHTTPRequestHandler):
    def _send_json(self, data, status_code=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw_data = self.rfile.read(length).decode("utf-8") if length > 0 else ""
            data = json.loads(raw_data) if raw_data else {}

            now_t = time.strftime("%H:%M:%S")
            print(f"[{now_t}] 收到主站推播數據:", flush=True)
            if "results" in data and isinstance(data["results"], list):
                for item in data["results"]:
                    tag = "🟢 [OK]" if "-OK" in item else "🔴 [NG]"
                    print(f"   {tag} {item}", flush=True)
            elif "result" in data:
                print(f"   {data['result']}", flush=True)
            else:
                print(f"   {data}", flush=True)

            self._send_json({"status": "received", "time": now_t})
        except Exception as e:
            print(f"解析失敗: {e}", flush=True)
            self._send_json({"error": str(e)}, status_code=400)

    def do_GET(self):
        self._send_json({"status": "receiver online", "listen_port": PORT})

    def log_message(self, format, *args):
        pass


class ThreadedServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def start_listener(port=PORT):
    server = ThreadedServer((HOST, port), ReceiverHandler)
    print("=" * 55, flush=True)
    print("      NYA HTTP 接收端 (Receiver Server)", flush=True)
    print("=" * 55, flush=True)
    print(f"監聽端口: http://0.0.0.0:{port}", flush=True)
    print(f"等待主站 (192.168.1.4:8080) 推送數據...\n", flush=True)
    server.serve_forever()


def start_polling(master_url=MASTER_URL, interval=0.5):
    print("=" * 55, flush=True)
    print("      NYA HTTP 輪詢客戶端 (Polling Client)", flush=True)
    print("=" * 55, flush=True)
    print(f"目標主站: {master_url}", flush=True)
    print(f"輪詢間隔: {interval}s\n", flush=True)

    last_latest = None
    while True:
        try:
            with urllib.request.urlopen(master_url, timeout=1.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                latest = data.get("latest", "NONE")
                if latest != "NONE" and latest != last_latest:
                    last_latest = latest
                    now_t = time.strftime("%H:%M:%S")
                    print(f"[{now_t}] 獲取主站結果: {latest} | 全部: {data.get('results', [])}", flush=True)
        except Exception:
            pass
        time.sleep(interval)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "poll":
        target = sys.argv[2] if len(sys.argv) > 2 else MASTER_URL
        start_polling(target)
    else:
        start_listener()
