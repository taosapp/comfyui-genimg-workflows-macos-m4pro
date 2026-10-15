#!/usr/bin/env bash
# ============================================================================
#  ComfyUI 启动器（macOS / Apple Silicon）
#
#  用法：
#    chmod +x start_comfyui.sh      # 从别处拷来后补一次执行位即可
#    ./start_comfyui.sh             # 正常启动
#    ./start_comfyui.sh --cpu       # 只验证节点、不加载模型（排查用）
#    ./start_comfyui.sh --lowvram   # 内存吃紧时手动降载
#
#  可用环境变量：
#    PYEXE         指定解释器（默认自动探测项目内 venv）
#    COMFY_ARGS    追加给 main.py 的参数
#    HF_ENDPOINT   HuggingFace 镜像（默认 hf-mirror.com）
#    HTTP_PROXY    走代理拉模型，例如 http://127.0.0.1:7897
#
#  关于硬件（本仓库适配机型：M4 Pro / 24GB 统一内存 / 无独显）：
#    · Apple Silicon 没有 CUDA，走 **MPS** 后端；
#    · 24GB 是 CPU/GPU **共享**内存，不是独立显存，所以：
#        - 不要加 --lowvram（那是 NVIDIA 显存不足时的策略，在 MPS 下反而更慢）；
#        - 默认让 ComfyUI 自己判断（normalvram），文本编码器算完提示词即卸载；
#    · 工作流里的 Qwen-Image GGUF 主模型走 ComfyUI-GGUF 节点（CPU 解量化后送 MPS），
#      这是目前 Apple Silicon 上跑 Qwen-Image 最稳的路径；
#    · MPS 不支持的算子（如 fp8/e4m3、部分 int8 量化路径）靠
#      PYTORCH_ENABLE_MPS_FALLBACK 回落到 CPU，不会直接崩，但会明显变慢。
# ============================================================================

set -uo pipefail

# ---- 0. 切到脚本所在目录（ComfyUI 根目录）--------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

if [ "$(uname -s)" != "Darwin" ]; then
    echo "[警告] 这份脚本按 macOS 调优；在 $(uname -s) 上 MPS 相关设置不会生效。"
fi

# ---- 1. 环境变量 -----------------------------------------------------------
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# MPS 缺算子时回落 CPU，而不是抛异常中断出图
export PYTORCH_ENABLE_MPS_FALLBACK=1
# 本机代理常把 127.0.0.1 也劫持走，访问本地服务要绕开
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="$NO_PROXY"

# 注意：不要设 PYTORCH_MPS_HIGH_WATERMARK_RATIO。
# 实测（torch 2.11.0）该变量任何非零值都会让 MPS 初始化直接抛
#   RuntimeError: invalid low watermark ratio
# 而设 0.0 与不设完全等价（recommended_max_memory 都是 17.76GB），属于 no-op。
# PYTORCH_MPS_FAST_MATH / PYTORCH_MPS_PREFER_METAL 也实测无收益（见 README），均不加。

# ---- 2. 定位解释器：优先 .venv，其次系统 python3 ---------------------------
PY="${PYEXE:-}"
if [ -z "$PY" ]; then
    for cand in ".venv/bin/python" "venv/bin/python" "env/bin/python"; do
        if [ -x "$cand" ]; then PY="$cand"; break; fi
    done
fi
if [ -z "$PY" ]; then
    PY="$(command -v python3 || command -v python || true)"
fi

echo "============================================================"
echo "  ComfyUI 启动（macOS / Apple Silicon）"
echo "============================================================"
echo "  目录 : $SCRIPT_DIR"

if [ -z "$PY" ]; then
    echo
    echo "[错误] 没找到可用的 Python。"
    echo "       先建环境：  python3 -m venv .venv"
    echo "       再装依赖：  .venv/bin/pip install -r requirements.txt"
    echo "       或直接跑一键脚本： python3 scripts/prepare.py"
    exit 1
fi

echo "  解释器: $PY"

if [ ! -f "$SCRIPT_DIR/main.py" ]; then
    echo
    echo "[错误] 当前目录没有 main.py —— 这不是 ComfyUI 根目录。"
    echo "       本仓库需要先补齐上游运行时："
    echo "         python3 scripts/prepare.py"
    exit 1
fi

# ---- 3. 打印环境摘要 -------------------------------------------------------
"$PY" - <<'PYEOF'
import platform, sys
print("  Python :", sys.version.split()[0], "|", platform.system(), platform.machine())

try:
    import torch
except Exception as exc:
    print("  [!] torch 导入失败：%r" % (exc,))
    print("      先装依赖： .venv/bin/pip install -r requirements.txt")
    sys.exit(0)

line = "  torch  : %s" % torch.__version__
try:
    if torch.backends.mps.is_available():
        line += " | MPS 可用: True"
    else:
        line += " | MPS 可用: False（会退回 CPU，出图很慢）"
    if torch.backends.mps.is_built():
        line += " (built)"
except Exception:
    pass
print(line)

# 统一内存容量：决定能不能塞下主模型 + 文本编码器
try:
    import subprocess
    out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
    if out.returncode == 0 and out.stdout.strip().isdigit():
        print("  统一内存: %.0f GB（MPS 与 CPU 共用，不是独立显存）"
              % (int(out.stdout.strip()) / 1024**3))
except Exception:
    pass

try:
    import numpy
    print("  numpy  :", numpy.__version__)
except Exception:
    pass
PYEOF

# ---- 4. 取本机局域网 IP（仅用于打印提示，不默认监听）-----------------------
LANIP=""
if command -v ipconfig >/dev/null 2>&1; then
    LANIP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true)"
fi

echo "------------------------------------------------------------"
echo "  本机访问   : http://127.0.0.1:8188"
if [ -n "$LANIP" ]; then
    echo "  局域网访问 : http://${LANIP}:8188"
fi
echo "  停止服务   : 在此窗口按 Ctrl+C"
echo "------------------------------------------------------------"
echo "  追加参数   : ${COMFY_ARGS:-（无）}"
echo

# ---- 5. 启动 ---------------------------------------------------------------
# 关键参数 --gpu-only：把文本编码器也放上 MPS。
#
# 为什么必须加：MPS 下 ComfyUI 强制 vram_state = VRAMState.SHARED（统一内存），
# 而 text_encoder_device() 只在 HIGH_VRAM/NORMAL_VRAM 时才返回 GPU ——
# SHARED 不在其中，所以**不加这个参数时，8.7GB 的文本编码器整个在 CPU 上跑**。
# 这是 Mac 上最隐蔽的性能陷阱，加上后首跑/改提示词的耗时才正常。
#
# 显存账（24GB 统一内存，峰值发生在文本编码阶段）：
#   文本编码器 8.7G + 主模型 GGUF 4.9G + Turbo LoRA 0.9G ≈ 14.5G
#   < MPS 软上限 17.76G（24G × 0.74），放得下。
#
# 如果真报 OOM（极少见），临时降载用：./start_comfyui.sh --lowvram
# （--gpu-only 与 --lowvram/--highvram 互斥，只能二选一）
DEFAULT_ARGS="--gpu-only"
# shellcheck disable=SC2086
exec "$PY" main.py $DEFAULT_ARGS ${COMFY_ARGS:-} "$@"