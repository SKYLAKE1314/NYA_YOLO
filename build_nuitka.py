"""
NYA AI Studio — Nuitka 打包腳本 v7 (完美解法版)
策略：
  1. Nuitka 正常打包所有依賴套件（不使用會導致依賴斷裂的 nofollow）
  2. 針對會導致 MSVC C1002 堆積溢出的「超巨型子模組」（動輒數十萬行 C 代碼），
     在編譯前自動將其替換為輕量級 Dummy 假檔（Mock）。
  3. 編譯完成後，自動將原始檔案還原。
  這樣既能完美避開 MSVC 崩潰，又能保證程式運行時不會出現 ImportError！
"""

import os
import sys
import shutil
import subprocess
import importlib.util

# ==========================================
# 定義需要被替換為 Dummy 的超巨型模組
# ==========================================
MOCK_MODULES = {
    # 解決 sympy 導致的 C1002 (26,000行 C代碼)
    "sympy.polys.polyquinticconst": "class PolyQuintic:\n    pass\n",
    # 解決 PyTorch 導致的 C1002 (739,430行 C代碼)
    "torch.testing._internal.common_methods_invocations": "# Dummy mock\n",
}

def get_module_file(module_name):
    try:
        spec = importlib.util.find_spec(module_name)
        if spec and spec.origin:
            return spec.origin
    except Exception:
        pass
    return None

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    ui_dir = os.path.join(root_dir, "UI")
    main_script = os.path.join(ui_dir, "NyaAIStudio.py") if os.path.exists(os.path.join(ui_dir, "NyaAIStudio.py")) else os.path.join(ui_dir, "NyaYOLOStudio.py")
    icon_path = os.path.join(ui_dir, "icon.ico")
    wallpaper_path = os.path.join(ui_dir, "file_0000000031e8720681bd49398eace5bf.png")
    config_creator = os.path.join(root_dir, "ConfigCreator.py")

    cpu_cores = max(1, (os.cpu_count() or 4) - 1)

    cmd = [
        sys.executable, "-m", "nuitka",
        "--standalone",
        f"--jobs={cpu_cores}",
        "--lto=no",
        "--enable-plugins=pyside6",
        "--module-parameter=torch-disable-jit=no",
        "--include-package=onnx",
        "--include-package=onnxruntime",
        "--include-package=onnxslim",
        "--include-package-data=ultralytics",
        "--include-package-data=clip",
        "--include-package-data=onnx",
        "--include-package-data=onnxruntime",
        "--include-package-data=onnxslim",
        "--no-deployment-flag=self-execution",
        "--company-name=SKYLAKE",
        "--product-name=NyaDLTT",
        "--output-filename=NyaDLTT.exe",
        "--file-description=Nya DeepLearning Train Platform",
        "--file-version=0.1.0.0",
        "--product-version=0.1.0.0",
        "--copyright=by SKYLAKE, MIT License",
        "--assume-yes-for-downloads",
        "--output-dir=dist_nuitka",
    ]

    # 圖標
    if os.path.exists(icon_path):
        cmd.append(f"--windows-icon-from-ico={icon_path}")
        cmd.append(f"--include-data-files={icon_path}=UI/icon.ico")

    # UI/icons/ 資料夾
    icons_dir = os.path.join(ui_dir, "icons")
    if os.path.exists(icons_dir) and os.listdir(icons_dir):
        cmd.append(f"--include-data-dir={icons_dir}=UI/icons")

    # 壁紙
    if os.path.exists(wallpaper_path):
        cmd.append(f"--include-data-files={wallpaper_path}=UI/file_0000000031e8720681bd49398eace5bf.png")

    # ConfigCreator.py
    if os.path.exists(config_creator):
        cmd.append(f"--include-data-files={config_creator}=ConfigCreator.py")

    cmd.append(main_script)

    print("🚀 開始執行 Nuitka 打包程序 (v7.01)...")

    # 1. 備份並替換巨型檔案
    backups = {}
    print("\n🛡️  正在攔截並替換會導致 MSVC 崩潰的巨型模組...")
    try:
        for mod, mock_content in MOCK_MODULES.items():
            file_path = get_module_file(mod)
            if file_path and os.path.exists(file_path):
                # 讀取備份
                with open(file_path, "r", encoding="utf-8") as f:
                    backups[file_path] = f.read()
                # 寫入 Mock
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(mock_content)
                print(f"    ✅ 已替換: {mod}")

        # 2. 執行編譯
        print(f"\n⚡ 已啟用 {cpu_cores} 核心多線程 C 編譯！首次編譯預計 15~25 分鐘")
        print("\n執行的指令:\n" + " \\\n    ".join(cmd) + "\n")
        subprocess.run(cmd, check=True, cwd=root_dir)
        print("\n🎉 打包成功！生成的可執行檔位於: dist_nuitka/NyaYOLOStudio.dist/NyaDLTT.exe")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ 打包過程發生錯誤 (返回碼: {e.returncode})")
    finally:
        # 3. 恢復原始檔案
        print("\n📦 正在還原原始模組檔案...")
        for file_path, original_content in backups.items():
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(original_content)
                print(f"    ✅ 已還原: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"    ⚠️ 還原失敗 {file_path}: {e}")

if __name__ == "__main__":
    main()
