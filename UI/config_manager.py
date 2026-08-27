"""
ConfigCacheManager — 全局 UI 設定快取管理器
負責儲存與自動恢復使用者在 NYA AI Studio 的各項輸入與超參數配置
"""

import os
import json

CONFIG_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_cache.json")


def load_ui_cache():
    """讀取 config_cache.json 快取資料"""
    if os.path.exists(CONFIG_CACHE_FILE):
        try:
            with open(CONFIG_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_ui_cache(data):
    """寫入 config_cache.json 快取資料"""
    try:
        with open(CONFIG_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed to save UI cache: {e}")
