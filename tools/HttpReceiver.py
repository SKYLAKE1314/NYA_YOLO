import sys
import time
import json
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8080/result"


def receive_results(url=DEFAULT_URL, interval=0.05):
    print("=" * 60, flush=True)
    print("      NYA AI Studio - HTTP 接收工具 (Client)", flush=True)
    print("=" * 60, flush=True)
    print(f"連線主站: {url}", flush=True)
    print(f"正在即時接收檢測結果...\n", flush=True)

    last_id = None
    last_latest = None

    while True:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "NYA-Receiver"})
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))

                cur_id = data.get("id")
                latest = data.get("latest", "NONE")

                is_new = False
                if cur_id is not None:
                    if cur_id != last_id:
                        last_id = cur_id
                        is_new = True
                elif latest != "NONE" and latest != last_latest:
                    last_latest = latest
                    is_new = True

                if is_new and latest != "NONE":
                    now_t = time.strftime("%H:%M:%S")
                    results = data.get("results", [latest])
                    for res in results:
                        tag = "🟢 [OK]" if "-OK" in res else "🔴 [NG]"
                        print(f"[{now_t}] 接收結果: {tag} {res}", flush=True)

        except Exception:
            pass
        time.sleep(interval)


if __name__ == "__main__":
    target_url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    receive_results(target_url)
