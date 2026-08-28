"""
=============================================================================
NYA AI Studio - ClassifyTool.exe 自動封裝打包腳本 (PyInstaller / Nuitka)
用於將 ClassifyTool 二分類即時主站獨立封裝為可分發的 Windows .exe 執行檔
=============================================================================
"""

import os
import sys
import shutil
import subprocess

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    source_script = os.path.join(root_dir, "Detection", "ClassifyTool.py")
    if not os.path.exists(source_script):
        source_script = os.path.join(root_dir, "ClassifyTool.py")

    icon_path = os.path.join(root_dir, "UI", "icon.ico")
    out_dir = os.path.join(root_dir, "dist_classify")

    print("=" * 65)
    print("      ClassifyTool.exe 二分類獨立服務端打包程序")
    print("=" * 65)
    print(f"📦 來源腳本: {source_script}")
    print(f"📂 輸出目錄: {out_dir}")

    # 檢查 PyInstaller 是否已安裝
    try:
        import PyInstaller
        has_pyinstaller = True
    except ImportError:
        has_pyinstaller = False

    if not has_pyinstaller:
        print("\n⚙ 正在安裝 PyInstaller...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)

    # 建立 PyInstaller 打包指令
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name=ClassifyTool",
        "--onedir",                  # 資料夾模式（執行速度快且穩定）
        "--console",                 # 保留主站終端機日誌窗口
        "--noconfirm",
        "--clean",
        f"--distpath={out_dir}",
        "--hidden-import=ultralytics",
        "--hidden-import=torch",
        "--hidden-import=cv2",
        "--hidden-import=yaml",
        "--hidden-import=numpy",
        "--hidden-import=PIL",
        "--hidden-import=http.server",
        "--collect-data=ultralytics",
    ]

    if os.path.exists(icon_path):
        cmd.append(f"--icon={icon_path}")

    cmd.append(source_script)

    print("\n🚀 開始執行打包編譯...")
    print("指令:", " ".join(cmd))
    res = subprocess.run(cmd, cwd=root_dir)

    if res.returncode == 0:
        exe_dir = os.path.join(out_dir, "ClassifyTool")
        
        # 自動在 exe 同層建立 《verify》 與 《weight》 資料夾
        os.makedirs(os.path.join(exe_dir, "verify"), exist_ok=True)
        os.makedirs(os.path.join(exe_dir, "weight"), exist_ok=True)
        
        # 複製最新模型到 weight/
        src_best = os.path.join(root_dir, "weight", "best.pt")
        if not os.path.exists(src_best):
            src_best = os.path.join(root_dir, "runs", "classify", "train-5", "weights", "best.pt")
        
        if os.path.exists(src_best):
            dst_best = os.path.join(exe_dir, "weight", "best.pt")
            shutil.copy2(src_best, dst_best)
            print(f"💾 已自動將模型權重複製至: {dst_best}")

        print("\n" + "=" * 65)
        print("  🎉 打包完全成功！")
        print(f"  執行檔位置: {os.path.join(exe_dir, 'ClassifyTool.exe')}")
        print(f"  監控目錄  : {os.path.join(exe_dir, 'verify')}")
        print(f"  權重目錄  : {os.path.join(exe_dir, 'weight')}")
        print("=" * 65)
    else:
        print("\n❌ 打包失敗，請檢查上方編譯錯誤日誌。")

if __name__ == "__main__":
    main()
