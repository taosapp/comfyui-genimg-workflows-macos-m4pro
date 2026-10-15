#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一键准备脚本：新机器 clone 本仓库后，用 **Python 3.10+** 跑一次即可。

    python3.13 scripts/prepare.py      # 完整准备（大模型下载前会询问）
    python3.13 scripts/prepare.py --check    # 只检查不安装/不下载
    python3.13 scripts/prepare.py --yes      # 下载模型不再逐个询问

⚠️ **必须是 3.10 及以上**，而且要用它来建 .venv（脚本用 `sys.executable` 建 venv，
所以你用哪个 Python 跑它，venv 就是哪个版本）。

    macOS 自带的 /usr/bin/python3 是 3.9.x，**装不了 torch**：
      · PyTorch 从未发布过 cp39 的 macOS arm64 wheel（实测：cp39 0 个 / cp313 6 个）
      · transformers / tokenizers / safetensors / huggingface_hub 要求 >=3.10
      · numpy 2.x 要求 >=3.12
    装一个并用即可（不影响系统自带那个）：
      brew install python@3.13
      python3.13 scripts/prepare.py

它做五件事：
  1. 运行环境：在项目内建 .venv（Apple Silicon / macOS 原生，不需要 conda）；
  2. 上游运行时：本仓库只跟踪策划资产，若 main.py 缺失则浅克隆官方 ComfyUI 补齐；
  3. 依赖：安装 PyTorch（MPS 后端）与 ComfyUI requirements.txt；
  4. 自定义节点：克隆 ComfyUI-GGUF 并打上 patches/ 里的 Qwen-Image 架构补丁；
  5. 模型：按下方清单检查 models/，缺失的可自动下载（支持断点续传）。

只用标准库，不要求预先装任何包。国内网络默认走 hf-mirror.com；
GitHub 拉不动时可用环境变量 HTTP_PROXY=http://127.0.0.1:7897 走本机代理。
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# ---- 版本守卫：必须在任何动作之前拦下 ---------------------------------------
# 放这里而不是等到 pip 装 torch 时才失败：3.9 的报错是
# "No matching distribution found for torch"，看不出是 Python 版本问题。
MIN_PY = (3, 10)
if sys.version_info < MIN_PY:
    sys.stderr.write(
        "\n[错误] 需要 Python %d.%d+，当前是 %s。\n"
        "       \n"
        "       macOS 自带的 /usr/bin/python3 是 3.9.x，装不了本项目的 PyTorch：\n"
        "         · PyTorch 没有发布过 cp39 的 macOS arm64 wheel\n"
        "         · transformers / tokenizers / safetensors / huggingface_hub 要求 >=3.10\n"
        "         · numpy 2.x 要求 >=3.12\n"
        "       \n"
        "       系统自带的那个不要动（归 root 且开了 SIP，改不了也删不掉）。\n"
        "       并装一个新版即可，两者互不干扰：\n"
        "         brew install python@3.13\n"
        "         python3.13 scripts/prepare.py\n"
        "\n" % (MIN_PY[0], MIN_PY[1], platform.python_version())
    )
    raise SystemExit(1)

ROOT = Path(__file__).resolve().parent.parent
HF_MIRROR = os.environ.get("HF_MIRROR", "https://hf-mirror.com").rstrip("/")
GGUF_URL = "https://github.com/city96/ComfyUI-GGUF"
UPSTREAM = "https://github.com/comfyanonymous/ComfyUI"

# 国内 PyPI 镜像（Apple Silicon 的 torch wheel 很大，走官方源会非常慢）
PIP_INDEX = os.environ.get("PIP_INDEX", "https://pypi.tuna.tsinghua.edu.cn/simple")
PIP_EXTRA = os.environ.get("PIP_EXTRA", "https://mirrors.aliyun.com/pypi/simple/")

# ---- 模型清单（文件名必须与工作流加载器里的一致，一字不差）-----------------
# url=None 的条目无法自动下载，脚本会给出获取指引。
MODELS = [
    # -- Qwen-Image-2.1 文生图 / 图片风格转换 / 中国彩墨 --
    dict(path="models/diffusion_models/qwen-image-2.1-Q5_K_M.gguf", size="4.9G",
         url=HF_MIRROR + "/unsloth/Qwen-Image-2.1-GGUF/resolve/main/qwen-image-2.1-Q5_K_M.gguf"),
    # 文本编码器：工作流默认指向 int8_convrot。峰值内存 = 主模型 4.9G + 文本编码器
    # 9.3G ≈ 14.2G，24GB 统一内存留足了余量（bf16 版是 17.5G，合计 22.4G，
    # 叠加 macOS 自身占用会直接把内存打满、开始 swap，出图慢到不可用）。
    dict(path="models/text_encoders/qwen3vl_8b_int8_convrot.safetensors", size="8.7G",
         url=HF_MIRROR + "/Comfy-Org/Qwen-Image-2.1/resolve/main/text_encoders/qwen3vl_8b_int8_convrot.safetensors",
         note="工作流默认用它。MPS 上 int8 量化算子可能不走 MPS 而回落 CPU"
              "（不崩，但慢）；若发现很慢，见 README「MPS 已知限制」"),
    # bf16 备选：MPS 原生精度、速度更好，但只有内存富余（如 36GB+）的机器才建议换。
    dict(path="models/text_encoders/qwen3vl_8b_bf16.safetensors", size="17.5G",
         url=HF_MIRROR + "/Comfy-Org/Qwen-Image-2.1/resolve/main/text_encoders/qwen3vl_8b_bf16.safetensors",
         optional=True,
         note="可选。内存 ≥36GB 时把它换上（改工作流 CLIPLoader 的 clip_name），"
              "MPS 原生 bf16 明显更快；24GB 机器请别用"),
    dict(path="models/vae/qwen_image_2.1_vae_bf16.safetensors", size="0.6G",
         url=HF_MIRROR + "/Comfy-Org/Qwen-Image-2.1/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors"),
    # -- Qwen-Image-Layered 分层文生图（可选，本机内存吃紧建议先跳过）--
    dict(path="models/diffusion_models/qwen-image-layered-Q4_K_S.gguf", size="12G",
         url=HF_MIRROR + "/unsloth/Qwen-Image-Layered-GGUF/resolve/main/qwen-image-layered-Q4_K_S.gguf",
         optional=True,
         note="分层文生图用。12G 主模型 + 8.7G 文本编码器，24GB 机器上比较勉强"),
    dict(path="models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", size="8.8G",
         url=None, optional=True,
         note="分层文生图用。fp8 在 MPS 上不受支持，需自己找 bf16 版本替换"),
    dict(path="models/vae/qwen_image_layered_vae.safetensors", size="0.3G",
         url=HF_MIRROR + "/Comfy-Org/Qwen-Image-Layered_ComfyUI/resolve/main/split_files/vae/qwen_image_layered_vae.safetensors",
         optional=True, note="分层文生图用"),
    # -- BiRefNet 抠图（体积小，强烈建议装）--
    dict(path="models/background_removal/birefnet.safetensors", size="0.4G",
         url=HF_MIRROR + "/Comfy-Org/BiRefNet/resolve/main/background_removal/birefnet.safetensors"),
    # -- Turbo 蒸馏 LoRA：彩墨极速版 / 图片风格转换极速版**必需**（缺它这两个工作流跑不了）--
    # 上游 Viggle/Qwen-Image-2.1-viggle-turbo（Qwen Research License，非商用）；
    # 已核对 SHA256 = 2bbf5b6c91d19ecb568a63d61ac490aa679a6a7bef14a68291abd386c0d4f77f
    dict(path="models/loras/qwen_image_2.1_viggle_turbo_v0.2.1_r128_comfy.safetensors", size="0.8G",
         url=HF_MIRROR + "/t8star/Zen-Image-Edit-Comfy/resolve/main/viggle-v0.2.1/"
                         "qwen_image_2.1_viggle_turbo_v0.2.1_r128_comfy.safetensors"),
]

ok = lambda s: print("  [OK]", s)
warn = lambda s: print("  [!!]", s)
do = lambda s: print("  [..]", s)


def run(cmd, **kw):
    print("  $", " ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], **kw)


def unified_memory_gb():
    """macOS 统一内存容量（GB）；非 macOS 返回 None。"""
    try:
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        if out.returncode == 0 and out.stdout.strip().isdigit():
            return int(out.stdout.strip()) / 1024**3
    except Exception:
        pass
    return None


# ---------------------------------------------------------------- 环境 ----
def venv_python():
    """项目内 venv 的 python 路径；不存在返回 None。"""
    p = ROOT / ".venv" / "bin" / "python"
    return p if p.exists() else None


def setup_env():
    existing = venv_python()
    if existing:
        ok("运行环境已存在: %s" % existing)
        return existing
    do("在项目内创建 .venv（Apple Silicon 原生，不需要 conda）")
    # venv 不带 pip 时（部分发行版拆包了 ensurepip）会失败，退回 --without-pip + 手动引导
    r = run([sys.executable, "-m", "venv", str(ROOT / ".venv")])
    if r.returncode != 0:
        warn("venv 创建失败，尝试 --without-pip")
        if run([sys.executable, "-m", "venv", "--without-pip", str(ROOT / ".venv")]).returncode != 0:
            warn("无法创建虚拟环境，请手动准备 Python 后重试")
            return None
    py = venv_python()
    if not py:
        warn(".venv 里没有 python 可执行文件")
        return None
    ok("已创建: %s" % py)
    return py


def pip_install(env_py, *args):
    """装包。走国内镜像（Apple Silicon 的 torch 有几百 MB，官方源会非常慢）。"""
    base = [env_py, "-m", "pip", "install",
            "--index-url", PIP_INDEX, "--extra-index-url", PIP_EXTRA]
    return run(base + list(args)).returncode == 0


# ---- torch 三件套版本对应表 ------------------------------------------------
# 必须钉死版本！不指定时 pip 会在错综复杂的版本区间里无限回溯，
# 表现为「静默卡死」十几分钟（venv 体积不涨、pip 缓存不涨、进程还活着）。
# 这张表是 Python 3.13 + macOS arm64 上实测可用的一组；换 Python 版本需重查：
#   查法：curl -s https://pypi.tuna.tsinghua.edu.cn/simple/torch/ | grep -o 'torch-[^-]*-cp3XX-cp3XX-macosx[^"]*arm64\.whl'
TORCH_TRIO = [
    ("torch", "2.11.0"),
    ("torchvision", "0.26.0"),
    ("torchaudio", "2.11.0"),
]


def prefetch_big_wheels(dest):
    """把大 wheel 先用 curl 拉到本地再装。

    为什么：pip 在本机会对每个包重试沙箱代理的 SSL（UNEXPECTED_EOF_WHILE_READING），
    装这三件套要 12 分钟；而 curl 直连 TUNA 实测 36 MB/s（77MB 的 torch 只要 2.2 秒）。
    """
    import re
    import urllib.request
    dest.mkdir(parents=True, exist_ok=True)
    saved = 0
    for pkg, ver in TORCH_TRIO:
        # 期望下限：torch ~77MB、其余几百 KB 以上；低于此说明是上次中断的残文件
        min_size = 60 * 1024 ** 2 if pkg == "torch" else 300 * 1024
        target = next((f for f in dest.iterdir() if f.name.startswith("%s-%s" % (pkg, ver))), None)
        if target and target.stat().st_size > min_size:
            ok("%s %s 已在本地（%.1f MB）" % (pkg, ver, target.stat().st_size / 1024**2))
            continue
        try:
            idx = urllib.request.urlopen("%s/simple/%s/" % (PIP_INDEX, pkg), timeout=30).read().decode()
        except Exception as exc:
            warn("拉取 %s 索引失败：%r" % (pkg, exc))
            continue
        # 只取非 free-threaded（cp3XX 而非 cp3XXt）的 macOS arm64 wheel
        cands = re.findall(
            r'href="([^"]*%s-%s-cp3\d+-cp3\d+-macosx[^"#]*?arm64\.whl)'
            % (re.escape(pkg), re.escape(ver)), idx)
        if not cands:
            warn("PyPI 上找不到 %s %s 的 macOS arm64 wheel，跳过预下载" % (pkg, ver))
            continue
        url = "https://pypi.tuna.tsinghua.edu.cn/" + cands[0].replace("../../", "")
        fn = dest / url.rsplit("/", 1)[-1]
        do("预下载 %s %s（%.0f MB 级）" % (pkg, ver, {"torch": 77, "torchvision": 2, "torchaudio": 1}[pkg]))
        try:
            urllib.request.urlretrieve(url, str(fn))
            saved += 1
        except Exception as exc:
            warn("预下载 %s 失败：%r（交给 pip 自己下）" % (pkg, exc))
    return saved


def setup_deps(env_py):
    do("升级 pip / setuptools / wheel")
    pip_install(env_py, "--upgrade", "pip", "setuptools", "wheel")

    wheels = ROOT / "downloads" / "wheels"
    prefetch_big_wheels(wheels)

    do("安装 PyTorch 三件套（版本已钉死：%s）"
       % "、".join("%s==%s" % t for t in TORCH_TRIO))
    trio = ["%s==%s" % t for t in TORCH_TRIO]
    cmd = [env_py, "-m", "pip", "install", *trio, "--index-url", PIP_INDEX]
    if wheels.exists():
        cmd += ["--find-links", str(wheels)]
    cmd += ["--progress-bar", "off"]
    if run(cmd).returncode != 0:
        warn("torch 安装失败。可手动指定版本重试，参考脚本里的 TORCH_TRIO 表。")

    req = ROOT / "requirements.txt"
    if req.exists():
        do("安装 ComfyUI requirements.txt")
        pip_install(env_py, "-r", str(req))
    else:
        warn("requirements.txt 不存在（先完成上游运行时步骤）")


# ---------------------------------------------------------- 上游运行时 ----
def setup_upstream():
    if (ROOT / "main.py").exists():
        ok("上游 ComfyUI 运行时已就位")
        return
    do("本仓库只含策划资产，开始浅克隆官方 ComfyUI 补齐运行时")
    tmp = ROOT / "_upstream_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    # git 会自动沿用 http_proxy/https_proxy 环境变量，国内直连 GitHub 慢时
    # 用 HTTP_PROXY=http://127.0.0.1:7897 跑本脚本即可
    if run(["git", "clone", "--depth", "1", UPSTREAM, str(tmp)]).returncode != 0:
        warn("克隆失败。若是网络问题，可设置 HTTP_PROXY=http://127.0.0.1:7897 后重跑")
        return
    # 合并上游运行时，但**绝不覆盖本仓库自有的文件**。
    # 上游 ComfyUI 根目录同样有 README.md / LICENSE / docs/ / custom_nodes/，
    # 无脑 move 会把本项目的内容覆盖掉（2026-10-15 实际踩过：README 与
    # MIT LICENSE 被上游 GPL 覆盖，custom_nodes 被套成 custom_nodes/custom_nodes）。
    # 规则：同名文件已存在 -> 跳过并提示；同名目录已存在 -> 逐项递归合并。
    ours = {"README.md", "LICENSE", "workflows", "styles", "patches",
            "scripts", "docs", "custom_nodes", "models", "start_comfyui.sh",
            ".workbuddy", ".git"}
    skipped, merged, added = [], 0, 0
    for item in sorted(tmp.iterdir()):
        name = item.name
        if name == ".git":
            continue  # 上游 .git 不需要；将来更新可重跑本步
        dst = ROOT / name
        if name in ours and dst.exists():
            if dst.is_dir() and item.is_dir():
                # 目录：递归合并子项，绝不整体替换
                for sub in sorted(item.iterdir()):
                    target = dst / sub.name
                    if not target.exists():
                        shutil.move(str(sub), str(target))
                        merged += 1
                skipped.append(name + "/（已合并子项）")
            else:
                skipped.append(name)
            continue
        shutil.move(str(item), str(dst))
        added += 1
    shutil.rmtree(tmp, ignore_errors=True)
    ok("上游运行时已就位（新增 %d 项，合并 %d 项）" % (added, merged))
    if skipped:
        warn("以下同名文件/目录已存在，**保留本项目版本未覆盖**：%s" % "、".join(skipped))


# ------------------------------------------------------------ GGUF 节点 ----
def setup_gguf(env_py):
    dst = ROOT / "custom_nodes" / "ComfyUI-GGUF"
    if not dst.exists():
        do("克隆 ComfyUI-GGUF")
        if run(["git", "clone", GGUF_URL, str(dst)]).returncode != 0:
            warn("克隆失败，可设置 HTTP_PROXY=http://127.0.0.1:7897 后重跑；补丁步骤跳过")
            return
    patch = ROOT / "patches" / "comfyui-gguf-add-qwen-image-arch.patch"
    # 注意：Windows 上 clone 下来的 convert.py 是 **CRLF** 行尾，而仓库里的 patch 是 LF，
    # 严格模式会报 "patch does not apply"。所以先试严格模式，失败再退到 --ignore-whitespace。
    applied = False
    for extra in ([], ["--ignore-whitespace"]):
        chk = run(["git", "-C", str(dst), "apply", "--check", *extra, str(patch)], capture_output=True)
        if chk.returncode == 0:
            if run(["git", "-C", str(dst), "apply", *extra, str(patch)]).returncode == 0:
                ok("已打上 Qwen-Image 架构补丁" + ("（宽松模式：源文件为 CRLF 行尾）" if extra else ""))
                applied = True
            break
    if not applied:
        rev = run(["git", "-C", str(dst), "apply", "--check", "-R", "--ignore-whitespace", str(patch)],
                  capture_output=True)
        ok("补丁已打过" if rev.returncode == 0 else "补丁不适用（上游代码变动？见 patches/README.md）")
    req = dst / "requirements.txt"
    if req.exists():
        do("安装 ComfyUI-GGUF requirements.txt")
        pip_install(env_py, "-r", str(req))


# ---------------------------------------------------------------- 模型 ----
def human(n):
    return "%.1fGB" % (n / 1024**3)


def check_models(include_optional=True):
    missing = []
    for m in MODELS:
        if not include_optional and m.get("optional"):
            continue
        p = ROOT / m["path"]
        if p.exists() and p.stat().st_size > 1024 * 1024:
            ok("%s (%s)" % (m["path"], human(p.stat().st_size)))
        else:
            missing.append(m)
            warn("缺失 %s (约 %s)%s"
                 % (m["path"], m["size"], "  [可选]" if m.get("optional") else ""))
    return missing


def download(m, always_yes):
    p = ROOT / m["path"]
    p.parent.mkdir(parents=True, exist_ok=True)
    if not m.get("url"):
        warn("无自动下载地址：%s\n      %s" % (m["path"], m.get("note", "")))
        return False
    if not always_yes:
        try:
            hint = " [可选]" if m.get("optional") else ""
            if input("      下载 %s%s (约 %s)? [y/N] " % (m["path"], hint, m["size"])).strip().lower() != "y":
                warn("跳过 %s" % m["path"])
                return False
        except EOFError:
            return False
    if m.get("note"):
        do("说明：%s" % m["note"])
    do("curl 下载（支持断点续传，中断后重跑本脚本可续）")
    cmd = ["curl", "-L", "-C", "-", "--retry", "3", "--retry-delay", "2", "-o", str(p), m["url"]]
    # 本机代理常把境外站点也劫持走，hf-mirror 是国内直连更快；
    # 除非显式设了 PROXY，否则绕过代理直连。
    if not os.environ.get("PROXY"):
        cmd += ["--noproxy", "*"]
    return run(cmd).returncode == 0 and p.exists()


# ---------------------------------------------------------------- 主流程 ----
def report_platform():
    print("== 平台 ==")
    print("  系统   : %s %s" % (platform.system(), platform.release()))
    print("  架构   : %s" % platform.machine())
    mem = unified_memory_gb()
    if mem:
        print("  统一内存: %.0f GB（CPU/GPU 共享，不是独立显存）" % mem)
    if platform.system() == "Darwin":
        print("  加速后端: MPS（Apple Silicon 无 CUDA）")
        print("  启动脚本会设 PYTORCH_ENABLE_MPS_FALLBACK=1，MPS 缺算子时回落 CPU 而不崩")
    print("  代理   : %s" % (os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy") or "（未设置，直连）"))


def main():
    ap = argparse.ArgumentParser(description="ComfyUI 工作流仓库一键准备（macOS / Apple Silicon）")
    ap.add_argument("--check", action="store_true", help="只检查，不安装不下载")
    ap.add_argument("--yes", action="store_true", help="下载模型不询问")
    ap.add_argument("--skip-models", action="store_true", help="跳过模型检查/下载")
    ap.add_argument("--core-only", action="store_true",
                    help="只检查必需模型（跳过分层文生图那 12GB+ 的大件）")
    args = ap.parse_args()

    print("== 项目准备（根目录 %s）==" % ROOT)
    report_platform()

    print("== 1/5 运行环境 ==")
    env_py = venv_python() if args.check else setup_env()
    if env_py is None:
        if args.check:
            warn("缺少 .venv —— 运行 python3 scripts/prepare.py 创建")
            return 1
        warn("没有可用的 Python 环境，后续步骤无法继续")
        return 1

    print("== 2/5 上游运行时 ==")
    setup_upstream()

    print("== 3/5 依赖 ==")
    if not args.check:
        setup_deps(env_py)
    else:
        ok("（--check 跳过安装）")

    print("== 4/5 ComfyUI-GGUF 节点与补丁 ==")
    if args.check:
        ok("已就位" if (ROOT / "custom_nodes/ComfyUI-GGUF").exists() else "缺失（--check 跳过安装）")
    else:
        setup_gguf(env_py)

    print("== 5/5 模型 ==")
    missing = check_models(include_optional=not args.core_only)
    if missing and not args.check and not args.skip_models:
        for m in missing:
            download(m, args.yes)
        missing = check_models(include_optional=not args.core_only)

    print("== 完成 ==")
    print("启动：./start_comfyui.sh    （首次请先 chmod +x start_comfyui.sh）")
    print("浏览器打开 http://127.0.0.1:8188 ，把 workflows/ 里的 json 拖进去")
    if missing:
        print("仍缺 %d 个模型文件，见上方 [!!] 清单。" % len(missing))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())