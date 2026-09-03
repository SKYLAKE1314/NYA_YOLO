"""
NYA AI STUDIO - Windows 原生 MSIX 安裝包建構腳本
支援自動偵測 Windows SDK、生成 Visual Assets 圖標、生成 AppxManifest、MakeAppx 打包、自簽署憑證與 SignTool 數位簽名
"""

import os
import sys
import glob
import json
import shutil
import argparse
import subprocess
from PIL import Image

VERSION = "1.0.0.0"
PUBLISHER = "CN=SKYLAKE"
PACKAGE_NAME = "NYAAIStudio"
DISPLAY_NAME = "NYA AI STUDIO"
DESCRIPTION = "NYA DeepLearning Train Platform"
EXECUTABLE_NAME = "NyaDLTT.exe"
CERT_PASSWORD = "SkylakePassword123"


def find_windows_sdk_tool(tool_name):
    """在系統中尋找 Windows Kits 10/11 SDK 工具（如 makeappx.exe, signtool.exe）"""
    # 1. 檢查 PATH
    in_path = shutil.which(tool_name)
    if in_path:
        return in_path

    # 2. 搜尋 Windows Kits 10
    base_kits = r"C:\Program Files (x86)\Windows Kits\10\bin"
    if os.path.exists(base_kits):
        pattern = os.path.join(base_kits, "*", "x64", tool_name)
        matches = glob.glob(pattern)
        if matches:
            # 排序取最新版本
            matches.sort(reverse=True)
            return matches[0]

    return None


def generate_assets(assets_dir, icon_path):
    """由 icon.ico 生成符合 MSIX 規範的各尺寸視覺資產 PNG"""
    os.makedirs(assets_dir, exist_ok=True)

    if not os.path.exists(icon_path):
        raise FileNotFoundError(f"找不到圖標檔案: {icon_path}")

    ico = Image.open(icon_path)

    # 尺寸規格 (檔名, 畫布尺寸, 圖標縮放佔比)
    specs = [
        ("Square44x44Logo.png", (44, 44), 0.85),
        ("Square150x150Logo.png", (150, 150), 0.8),
        ("Wide310x150Logo.png", (310, 150), 0.8),
        ("Square310x310Logo.png", (310, 310), 0.8),
        ("StoreLogo.png", (50, 50), 0.85),
    ]

    for filename, canvas_size, scale in specs:
        out_path = os.path.join(assets_dir, filename)
        cw, ch = canvas_size

        # 計算圖標尺寸
        target_h = int(ch * scale)
        target_w = int(cw * scale)
        target_size = min(target_w, target_h)

        # 縮放圖標
        resized_ico = ico.resize((target_size, target_size), Image.Resampling.LANCZOS)
        if resized_ico.mode != "RGBA":
            resized_ico = resized_ico.convert("RGBA")

        # 建立透明畫布並置中黏貼
        canvas = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
        offset_x = (cw - target_size) // 2
        offset_y = (ch - target_size) // 2
        canvas.paste(resized_ico, (offset_x, offset_y), resized_ico)

        canvas.save(out_path, format="PNG")
        print(f"  [Asset] 生成 {filename} ({cw}x{ch})")


def generate_manifest(dist_dir, version=VERSION, publisher=PUBLISHER):
    """動態生成符合 Windows 10/11 Desktop Bridge 的 AppxManifest.xml"""
    manifest_path = os.path.join(dist_dir, "AppxManifest.xml")

    manifest_content = f"""<?xml version="1.0" encoding="utf-8"?>
<Package
  xmlns="http://schemas.microsoft.com/appx/manifest/foundation/windows10"
  xmlns:uap="http://schemas.microsoft.com/appx/manifest/uap/windows10"
  xmlns:rescap="http://schemas.microsoft.com/appx/manifest/foundation/windows10/restrictedcapabilities"
  IgnorableNamespaces="uap rescap">

  <Identity
    Name="{PACKAGE_NAME}"
    Publisher="{publisher}"
    Version="{version}"
    ProcessorArchitecture="x64" />

  <Properties>
    <DisplayName>{DISPLAY_NAME}</DisplayName>
    <PublisherDisplayName>SKYLAKE</PublisherDisplayName>
    <Logo>Assets\\StoreLogo.png</Logo>
  </Properties>

  <Dependencies>
    <TargetDeviceFamily Name="Windows.Desktop" MinVersion="10.0.17763.0" MaxVersionTested="10.0.26100.0" />
  </Dependencies>

  <Resources>
    <Resource Language="zh-TW" />
    <Resource Language="en-US" />
  </Resources>

  <Applications>
    <Application Id="{PACKAGE_NAME}"
      Executable="{EXECUTABLE_NAME}"
      EntryPoint="Windows.FullTrustApplication">
      <uap:VisualElements
        DisplayName="{DISPLAY_NAME}"
        Description="{DESCRIPTION}"
        BackgroundColor="transparent"
        Square150x150Logo="Assets\\Square150x150Logo.png"
        Square44x44Logo="Assets\\Square44x44Logo.png">
        <uap:DefaultTile
          Wide310x150Logo="Assets\\Wide310x150Logo.png"
          Square310x310Logo="Assets\\Square310x310Logo.png" />
      </uap:VisualElements>
    </Application>
  </Applications>

  <Capabilities>
    <rescap:Capability Name="runFullTrust" />
  </Capabilities>
</Package>
"""
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(manifest_content)
    print(f"  [Manifest] 已生成 AppxManifest.xml (版本: {version}, 發行者: {publisher})")
    return manifest_path


def ensure_certificate(cert_pfx_path, cert_cer_path, publisher=PUBLISHER):
    """若尚未存在憑證，則透過 PowerShell 自動生成代碼簽署憑證並匯出"""
    if os.path.exists(cert_pfx_path) and os.path.exists(cert_cer_path):
        print(f"  [Cert] 沿用現有代碼簽署憑證: {cert_pfx_path}")
        return True

    print("  [Cert] 正在生成自簽署代碼簽署憑證...")
    ps_content = f"""
$cert = New-SelfSignedCertificate -Type Custom -Subject '{publisher}' -KeyUsage DigitalSignature -FriendlyName 'NYA AI Studio Certificate' -CertStoreLocation 'Cert:\\CurrentUser\\My' -TextExtension @('2.5.29.37={{text}}1.3.6.1.5.5.7.3.3', '2.5.29.19={{text}}')
$pw = ConvertTo-SecureString -String '{CERT_PASSWORD}' -Force -AsPlainText
Export-PfxCertificate -Cert $cert -FilePath '{cert_pfx_path}' -Password $pw | Out-Null
Export-Certificate -Cert $cert -FilePath '{cert_cer_path}' | Out-Null
"""
    temp_ps1 = os.path.join(os.path.dirname(cert_pfx_path), "_temp_gen_cert.ps1")
    with open(temp_ps1, "w", encoding="utf-8") as f:
        f.write(ps_content)

    ps_exe = shutil.which("pwsh") or shutil.which("powershell") or "powershell.exe"
    ret = subprocess.run([ps_exe, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", temp_ps1], capture_output=True, text=True)

    if os.path.exists(temp_ps1):
        try:
            os.remove(temp_ps1)
        except Exception:
            pass

    if ret.returncode != 0 or not os.path.exists(cert_pfx_path):
        print(f"憑證生成失敗: {ret.stderr}\n{ret.stdout}")
        return False

    print(f"  [Cert] 憑證生成成功: {cert_cer_path}")
    return True


def sign_msix(signtool_path, msix_path, cert_pfx_path):
    """調用 SignTool.exe 為 MSIX 進行 SHA256 數位簽名"""
    cmd = [
        signtool_path, "sign",
        "/fd", "SHA256",
        "/a",
        "/f", cert_pfx_path,
        "/p", CERT_PASSWORD,
        msix_path
    ]
    ret = subprocess.run(cmd, capture_output=True, text=True)
    if ret.returncode != 0:
        print(f"MSIX 簽名失敗: {ret.stderr}\n{ret.stdout}")
        return False
    print(f"  [Sign] MSIX 數位簽章成功！(SHA256 / {PUBLISHER})")
    return True


def create_installer_bat(dist_msix_dir, msix_filename, cer_filename):
    """在 dist_msix 目錄下生成一鍵安裝批次檔"""
    bat_path = os.path.join(dist_msix_dir, "Install_MSIX.bat")
    content = f"""@echo off
chcp 65001 >nul
title NYA AI STUDIO - MSIX 安裝精靈
cd /d "%~dp0"

echo ========================================================
echo         NYA AI STUDIO - MSIX 快速安裝精靈
echo ========================================================
echo.

:: 檢查系統管理員權限
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [提示] 正在請求系統管理員權限以信任代碼簽名憑證...
    powershell -Command "Start-Process cmd -ArgumentList '/c \"\"%~f0\"\"' -Verb RunAs"
    exit /b
)

echo [1/2] 正在匯入代碼簽署憑證至受信任的人員 (TrustedPeople)...
certutil -addstore -f "TrustedPeople" "{cer_filename}" >nul 2>&1
if %errorLevel% equ 0 (
    echo       [成功] 憑證已成功信任！
) else (
    echo       [提示] 憑證匯入完成。
)

echo.
echo [2/2] 正在啟動 Windows MSIX 原生安裝精靈...
start "" "{msix_filename}"

echo.
echo 安裝介面已開啟，請在安裝視窗中點擊「安裝」即可完成部署！
timeout /t 6 >nul
exit /b
"""
    with open(bat_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  [Helper] 已生成一鍵安裝批次檔: {bat_path}")


def main():
    parser = argparse.ArgumentParser(description="Build MSIX package for NYA AI Studio")
    parser.add_argument("--version", default=VERSION, help="Package version (default: 1.0.0.0)")
    parser.add_argument("--dist-dir", default=None, help="Path to Nuitka compiled dist folder")
    parser.add_argument("--output-dir", default=None, help="Output directory for MSIX")
    parser.add_argument("--skip-pack", action="store_true", help="Skip MakeAppx packing if MSIX already exists")
    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.abspath(__file__))
    dist_dir = args.dist_dir or os.path.join(root_dir, "dist_nuitka", "NyaAIStudio.dist")
    out_dir = args.output_dir or os.path.join(root_dir, "dist_msix")
    icon_path = os.path.join(root_dir, "UI", "icon.ico")

    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("      NYA AI STUDIO - MSIX 原生安裝套件建構系統")
    print("=" * 60)

    # 1. 檢測 Windows SDK 工具
    makeappx_exe = find_windows_sdk_tool("makeappx.exe")
    signtool_exe = find_windows_sdk_tool("signtool.exe")

    if not makeappx_exe:
        print("[錯誤] 找不到 makeappx.exe！請確認是否已安裝 Windows 10/11 SDK。")
        sys.exit(1)
    if not signtool_exe:
        print("[錯誤] 找不到 signtool.exe！請確認是否已安裝 Windows 10/11 SDK。")
        sys.exit(1)

    print(f"[1/6] 偵測到 Windows SDK 工具鏈:")
    print(f"      MakeAppx: {makeappx_exe}")
    print(f"      SignTool: {signtool_exe}")

    # 2. 檢測 Nuitka 編譯輸出目錄
    exe_path = os.path.join(dist_dir, EXECUTABLE_NAME)
    if not os.path.exists(exe_path):
        print(f"[錯誤] 找不到編譯產物: {exe_path}")
        print("請先執行 python build_nuitka.py 完成獨立程式編譯後再執行本腳本！")
        sys.exit(1)
    print(f"[2/6] 來源程式確認就緒: {exe_path}")

    # 3. 生成 Visual Assets 圖標
    print("[3/6] 正在生成 MSIX 視覺資產圖標 (Assets)...")
    assets_dir = os.path.join(dist_dir, "Assets")
    generate_assets(assets_dir, icon_path)

    # 4. 生成 AppxManifest.xml
    print("[4/6] 正在生成 AppxManifest.xml 清單...")
    generate_manifest(dist_dir, version=args.version)

    # 5. 執行 MakeAppx 打包
    msix_filename = f"NyaAIStudio_{args.version}_x64.msix"
    out_msix_path = os.path.join(out_dir, msix_filename)

    if args.skip_pack and os.path.exists(out_msix_path) and os.path.getsize(out_msix_path) > 1024 * 1024:
        print(f"[5/6] 偵測到已有 MSIX 封裝檔 ({os.path.getsize(out_msix_path) / (1024*1024):.2f} MB)，跳過重複壓縮打包。")
    else:
        print(f"[5/6] 正在封裝為 MSIX 安裝檔: {out_msix_path} ...")
        cmd_pack = [makeappx_exe, "pack", "/d", dist_dir, "/p", out_msix_path, "/o"]
        res_pack = subprocess.run(cmd_pack, capture_output=True, text=True)
        if res_pack.returncode != 0:
            print(f"[錯誤] MakeAppx 打包失敗:\n{res_pack.stderr}\n{res_pack.stdout}")
            sys.exit(1)
        print("      [成功] MakeAppx 封裝完成！")

    # 6. 數位簽章與輔助工具
    print("[6/6] 正在進行數位簽章與安裝工具生成...")
    cert_pfx = os.path.join(out_dir, "NyaAIStudio.pfx")
    cert_cer = os.path.join(out_dir, "NyaAIStudio.cer")

    if not ensure_certificate(cert_pfx, cert_cer, publisher=PUBLISHER):
        sys.exit(1)

    if not sign_msix(signtool_exe, out_msix_path, cert_pfx):
        sys.exit(1)

    create_installer_bat(out_dir, msix_filename, "NyaAIStudio.cer")

    # 輸出完成摘要
    msix_size_mb = os.path.getsize(out_msix_path) / (1024 * 1024)
    print("\n" + "=" * 60)
    print("      MSIX 原生安裝套件建構完成！")
    print("=" * 60)
    print(f"MSIX 檔案: {out_msix_path} ({msix_size_mb:.2f} MB)")
    print(f"公鑰憑證: {cert_cer}")
    print(f"安裝批次: {os.path.join(out_dir, 'Install_MSIX.bat')}")
    print("\n使用說明:")
    print("  - 若本機或其他電腦欲安裝，直接執行 Install_MSIX.bat 即可一鍵信任並開啟原生安裝精靈。")
    print("  - 亦可手動雙擊 NyaAIStudio.cer 匯入至「受信任的人員」，之後直接雙擊 .msix 檔案安裝！")
    print("=" * 60)


if __name__ == "__main__":
    main()
