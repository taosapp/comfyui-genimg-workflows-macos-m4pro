# ComfyUI 图生图工作流集（Apple Silicon / macOS）

一套 ComfyUI 图片工作流，围绕 **Qwen-Image-2.1（GGUF 量化）图生图** 构建。
工作流本身与平台无关（同一份 JSON），本仓库已适配 **Apple Silicon（MPS，无独立显卡）**：

| 工作流 | 用途 | 速度（M4 Pro 实测） |
|---|---|---|
| `workflows/图片风格转换.json` | 通用图生图风格转换，下拉框切换风格库（水彩/简笔画/高清增强等），也可选「自定义」手写完整正/负向提示词。**纯极速版**（8 步 + Turbo LoRA / cfg 1.0） | 约 144 秒/张（720 档实测） |
| `workflows/抠图.json` | 透明底抠图（BiRefNet 专用分割模型，不改像素） | 待实测（轻量分割模型，预计十秒级） |
| `workflows/文生图.json` | Qwen-Image-2.1 文生图（GGUF + KV 缓存加速，质量档 20 步 / cfg 3.5） | 约 459 秒/张（720 档实测） |
| `workflows/文生图-极速.json` | 同上挂 Turbo 蒸馏 LoRA（8 步 / cfg 1.0）；**参考图接线 = 图生图，静音 = 纯文生图** | 约 117 秒/张（720 纯文生图实测），接参考图约 144 秒 |
| `workflows/分层文生图.json` | Qwen-Image-Layered 一次生成分层图（文字直接生成多层，LatentCut 切层输出） | 待实测 |
| `workflows/中国彩墨图生图.json` | 中国彩墨画，文生图 / 照片转彩墨画两用（默认画布 864×1152 / 3:4，通用提示词不绑定具体照片），质量档（26 步 / cfg 3.2） | 待实测（参考：720 档 20 步约 459 秒） |
| `workflows/中国彩墨图生图-极速.json` | 同上，挂 Turbo 蒸馏 LoRA（8 步 / cfg 1.0），**日常出图首选**（更快且更贴原照） | 待实测（参考：720 档 8 步约 117 秒） |

配套一个自建小节点包 `custom_nodes/comfyui-style-prompt/`（风格库下拉选择器 + 存回节点 + 背景填充节点），
以及一份 ComfyUI-GGUF 补丁（见 `patches/`）。

> 速度列为本机（**M4 Pro / 24GB 统一内存 / macOS**，MPS 后端）实测。速度预期见「MPS 已知限制」。
> 下方所有工作流文件、节点代码、Python 脚本均无需改动即可在 Mac 上运行。

## 工作流简介

### 图片风格转换
`LoadImage → 风格选择器(下拉) → TextEncodeQwenImage21 → KSampler → VAEDecode → SaveImage`，
中间串了一个 `QwenImage21Cache`（KV 缓存，同样提示词连续出图时可省 40% 时间，勿删）。

**纯极速版**（2026-10-15 精简）：链路里串了 Turbo 蒸馏 LoRA（⑯），8 步 / cfg 1.0，
M4 Pro 实测约 144 秒/张（720 档）。工作流内不再保留质量档；确需负向提示词生效的场合，
用 `中国彩墨图生图.json`（质量版，26 步 / cfg 3.2）的链路。

> ⚠️ cfg 1.0 下**负向提示词不参与计算**（`comfy/samplers.py:610`：cfg=1.0 时负向
> 条件被直接置 None，UNet 只算正向——这也是速度快的原因）。想约束画面（"别上色"、
> "别改脸"）**必须写进正向**：`02-简笔画` 即范例，黑白线稿约束全写在正向，
> 实测可出标准线稿；长指令反而压不住颜色，风格文件请保持极简。
> `03-hd_img` 的防改脸负向在本工作流里保护力弱。Turbo LoRA 许可为 Qwen Research
> License（非商用）。

风格 = `styles/` 目录下一个 `.txt` 文件，文件名（去掉序号）就是下拉框里的名字。
**新增风格只要丢一个 txt 进去、浏览器 F5 即可**；改内容直接重新 Queue。
风格文件支持 `#` 注释（详见 `styles/_说明.txt`）。

下拉框第一项是 **「自定义（只用下面两个框）」**：不读风格库，`describe` 框 = 完整正向提示词、
`avoid` 框 = 完整负向提示词，原样送进模型（前面不加任何库内容）。适合手写一次性提示词，
不用为了跑一次而新建风格文件。

> ⚠️ 工作流里的 `QwenImage21Cache` **`dtype` 必须保持 `default`**：改成实验性的 `int8`
> 在本机会在采样第一步抛 `aimdo memory compile error`（KV 缓存量化与动态显存不兼容）。

### 抠图
核心节点 `LoadBackgroundRemovalModel + RemoveBackground`（BiRefNet，Swin-L 版），
输出与输入同尺寸的透明 PNG。注意链路中必须过一次 `InvertMask`
（`JoinImageWithAlpha` 内部做 `alpha = 1 - mask`），`GrowMask` 放在反相前用于去白边。

### 文生图（质量版 / 极速版）
`UnetLoaderGGUF →（仅极速档）Turbo LoRA → QwenImage21Cache → TextEncodeQwenImage21 → KSampler → SaveImage`，
分辨率走 512 / 720 / 1024 下拉档。

- **极速版**（8 步 / cfg 1.0）720 档实测 **117 秒/张**，且实测光影更自然；「参考图（可选）」节点
  **线上接着就是图生图**（出图比例跟随参考图，约 144 秒），选中按 `M` 静音就回到纯文生图，不用改连线。
  接参考图时必须保留「VAELoader → 编码器.vae」这根线，否则图生图失效。
  > 📌 **换参考图**：点 ④ 节点里的「Choose File」上传，或把图片文件放进 `input/` 再在下拉里选。
  > 所有工作流的 LoadImage 默认指向 `example.png`（ComfyUI 自带，确保文件存在不会报错），
  > 换成自己的照片即可。文件名必须与 `input/` 里的实际文件一字不差。
- **质量版**（20 步 / cfg 3.5）实测约 **459 秒/张**，留作对照与正式交付（Turbo LoRA 是
  Qwen Research License **非商用**）。

> ⚠️ 极速档同样是 cfg 1.0，**负向提示词不参与计算**，要约束的内容全部写进正向（同「图片风格转换」）。

### 中国彩墨图生图（质量版 / 极速版）
同一套「空 latent + 参考图条件」链路的两个档位，提示词模板通用、不绑定具体照片
（换参考照片记得把画布改成同比例，如 9:16 → 864×1536）：

- **极速版**挂 Qwen-Image-2.1 Viggle Turbo 蒸馏 LoRA（8 步 / cfg 1.0），速度 M4 Pro 待实测
  （参考：720 档 8 步约 117 秒），且实测结构保真/纹理密度更好（结构 0.831~0.887 vs 0.534，纹理密度 13.0 vs 11.1），
  **日常出图首选**。首次运行慢约 3 分钟（LoRA 加载/修补一次性开销），服务不重启就一直快。
  > 🚫 **本文件已删除负向提示词框**：cfg 1.0 下负向完全不参与计算（`comfy/samplers.py:610`
  > 把 uncond 置 None），留着只会误导。做法是把 `negative_prompt` 接到一个恒空串的
  > Text 节点上——widget 被连线驱动后，那个文本框就从节点界面上消失了
  > （与 `图片风格转换.json` 同一思路）。原负向里的**画质约束**（印章/水印/模糊/噪点/杂乱构图…）
  > 已改写成正向表述并进提示词框；原负向里的**风格规避词**（照片/水彩/卡通/3D 渲染…）已丢弃——
  > 那些词放进正向会起反作用（等于命令模型去画水彩），且实测即使在负向里也是负资产。
- **质量版**（26 步 / cfg 3.2）保留作对照、以及需要用负向提示词控制画风的场合
  （本文件负向框仍在，cfg 3.2 下负向确实参与计算）。

> ⚠️ 本链路中负向提示词是「负资产」：负向里的「照片/摄影/写实」会把画面推离参考图结构
> （同种子实测结构保真 0.826 → 0.534）。cfg 1.0 时负向不参与计算，反而更贴原照。
> Turbo LoRA 许可为 Qwen Research License（**非商用**）。

### MiniMax-H3 图生视频（2026-10-15 已移除）
该工作流与配套的约 27GB 视频模型（unet 主模型 / 32B 文本编码器 / turbo LoRA / 音视频 VAE）
已从本仓库与本机一并移除，`scripts/prepare.py` 里的下载条目与 `--with-video` 参数同步删除。
需要时从 git 历史取回工作流 JSON，并按 `README` 旧版清单自行下载模型。

## 快速开始（新电脑）

本仓库 clone 下来后**只含策划资产**（工作流/风格库/自建节点/补丁/启动器），
运行时、依赖、模型都不在仓库里。用 **Python 3.10+** 跑一次准备脚本即可补齐：

```bash
git clone https://github.com/taosapp/comfyui-genimg-workflows-macos-m4pro.git
cd comfyui-genimg-workflows-macos-m4pro

# ⚠️ 需要 Python 3.10+，且必须用它来跑（脚本按 sys.executable 建 .venv）。
#    macOS 自带的 /usr/bin/python3 是 3.9.x，装不了 torch，先并装一个：
brew install python@3.13

python3.13 scripts/prepare.py                # 全自动：环境 + 运行时 + 依赖 + 节点补丁 + 模型下载
python3.13 scripts/prepare.py --check        # 或先只做检查，缺什么会列出来
python3.13 scripts/prepare.py --core-only    # 只装核心模型（跳过分层文生图那 20+GB）
```

> **为什么不能用系统自带的 Python**：PyTorch 从未发布过 cp39 的 macOS arm64 wheel
> （实测 cp39 0 个 / cp313 6 个）；transformers / tokenizers / safetensors / huggingface_hub
> 要求 >=3.10，numpy 2.x 要求 >=3.12。系统那个归 root 且开了 SIP，**不要动**，并装即可。


准备脚本会依次：在项目内建 `.venv`（Apple Silicon 原生，不需要 conda）→
浅克隆官方 ComfyUI 补齐运行时 → 装 PyTorch（MPS）与 ComfyUI 依赖（走清华镜像）→
克隆 ComfyUI-GGUF 并打上本仓库的补丁 → 按内置清单从 hf-mirror 下载缺失模型
（支持断点续传，中断重跑即可续）。

装完启动：

```bash
chmod +x start_comfyui.sh    # 首次补一次执行位
./start_comfyui.sh
```

浏览器打开 <http://127.0.0.1:8188>，把 `workflows/` 里的 json 拖进去即可。
少数模型没有自动下载地址（脚本会打印获取指引），按提示手动放置即可。

> **GitHub 拉不动时**（国内直连慢）：`HTTP_PROXY=http://127.0.0.1:7897 python3 scripts/prepare.py`。
> 该代理只影响 git clone；模型下载走 hf-mirror，脚本默认绕过代理直连（更快）。

## 安装

1. 安装 [ComfyUI](https://github.com/comfyanonymous/ComfyUI)（本仓库工作流基于 0.37.0 调好）；
2. 安装第三方节点 [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)：
   ```bash
   git clone https://github.com/city96/ComfyUI-GGUF custom_nodes/ComfyUI-GGUF
   .venv/bin/pip install -r custom_nodes/ComfyUI-GGUF/requirements.txt
   ```
3. **给 ComfyUI-GGUF 打补丁**（重要）：
   `qwen-image-2.1-Q5_K_M.gguf` 没有架构元数据，加载器只能从张量名反推架构，
   原版 ComfyUI-GGUF 不认识它、会报 `Unknown model architecture!`。
   打法见 `patches/README.md`（基线：city96/ComfyUI-GGUF @ `6ea2651e`）；
4. 把 `custom_nodes/comfyui-style-prompt/` 拷进你的 `custom_nodes/`，重启 ComfyUI；
5. 把 `workflows/`、`styles/` 放到 ComfyUI 根目录同名文件夹下；
6. 按下表下载模型（**模型不随仓库分发，请自行获取**）。

> 以上步骤 `scripts/prepare.py` 都会自动完成，手工做仅在需要定制时使用。

## 模型清单（图片工作流部分）

| 文件 | 大小 | 放置目录 | 来源 |
|---|---|---|---|
| `qwen-image-2.1-Q5_K_M.gguf` | 4.9 GB | `models/diffusion_models/` | [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) 官方权重的 GGUF 量化版（[Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1)、unsloth 等社区仓库有量化版） |
| `qwen3vl_8b_int8_convrot.safetensors` | 8.7 GB | `models/text_encoders/` | 基于 [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) 的 int8 convrot 量化版（ComfyUI 社区量化）。**Apple Silicon 默认用它**（峰值内存更省，见硬件要求）；内存 ≥36GB 可换官方 `qwen3vl_8b_bf16.safetensors`（17.5GB）换速度 |
| `qwen_image_2.1_vae_bf16.safetensors` | 0.63 GB | `models/vae/` | [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) |
| `qwen_image_2.1_viggle_turbo_v0.2.1_r128_comfy.safetensors`（**所有极速档必需**：图片风格转换 / 文生图-极速 / 彩墨极速） | 0.84 GB | `models/loras/` | Viggle Turbo v0.2.1 r128 蒸馏 LoRA（[Viggle/Qwen-Image-2.1-viggle-turbo](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo)，Qwen Research License **非商用**；本仓库实测 SHA256 `2bbf5b6c…4f77f`） |
| `qwen-image-layered-Q4_K_S.gguf`（分层文生图用） | 12 GB | `models/diffusion_models/` | Qwen-Image-Layered 的 GGUF 量化版（unsloth 等社区仓库） |
| `qwen_2.5_vl_7b_fp8_scaled.safetensors`（分层文生图用） | 8.7 GB | `models/text_encoders/` | Comfy-Org 社区重打包版（HF 搜 `qwen_2.5_vl_7b_fp8_scaled`） |
| `qwen_image_layered_vae.safetensors`（分层文生图用） | 0.24 GB | `models/vae/` | [Comfy-Org/Qwen-Image-Layered_ComfyUI](https://huggingface.co/Comfy-Org/Qwen-Image-Layered_ComfyUI) |
| `birefnet.safetensors`（仅抠图用） | 0.42 GB | `models/background_removal/` | [Comfy-Org/BiRefNet](https://huggingface.co/Comfy-Org/BiRefNet)（MIT） |

- 合计约 **36 GB**（其中分层文生图三件 21 GB，只跑风格转换/彩墨/抠图约 15.5 GB）；
  国内网络可用 [hf-mirror.com](https://hf-mirror.com) 镜像下载；`scripts/prepare.py` 内置本清单可自动下载；
- **文件名必须一字不差**：工作流按文件名找模型；
- 各模型许可证不同（Qwen 系列为 Apache-2.0 / Qwen Research，请自行确认你的用途是否合规）。

## 硬件要求

### Apple Silicon（当前适配目标）

| 配置 | 结论 |
|---|---|
| **M4 Pro / 24GB 统一内存（本机）** | 核心工作流（风格转换 / 彩墨 / 抠图 / 文生图）峰值内存约 **14.2GB**，24GB 够用。启动用默认参数，**不要** `--lowvram` |
| M 系列 16GB 统一内存 | 勉强，建议先用 `--core-only` 装模型（跳过分层文生图），并关掉其他吃内存的应用 |
| 内存 ≥36GB | 可把文本编码器换成 `qwen3vl_8b_bf16.safetensors`，MPS 原生 bf16 明显更快 |

**关键认知：24GB 是 CPU/GPU 共享的，不是独立显存。** 峰值 ≈ 主模型 4.9G + 文本编码器 8.7G +
LoRA/VAE ≈ **14.2GB**，剩下给系统。所以文本编码器**默认用 int8_convrot 而非 bf16**——
bf16 是 17.5G，合计 22.4G 会把 24GB 打满并触发 swap，反而慢到不可用。

启动：

```bash
./start_comfyui.sh                 # 默认参数（MPS，共享内存下这是对的）
./start_comfyui.sh --lowvram       # 仅在内存告急时手动降载
./start_comfyui.sh --cpu           # 只验证节点、不加载模型（排查用）
```

`start_comfyui.sh` 会自动做三件事，无需手动设置环境变量：

| 自动设置 | 作用 |
|---|---|
| `PYTORCH_ENABLE_MPS_FALLBACK=1` | MPS 不支持的算子回落 CPU，**不崩**（代价是慢） |
| `--gpu-only` | **把文本编码器也放上 MPS**，见下方「最大的性能陷阱」 |
| `NO_PROXY=127.0.0.1,localhost` | 本机代理常劫持回环地址，导致打不开 8188 |

> ⚠️ **不要**给 `PYTORCH_MPS_HIGH_WATERMARK_RATIO` 设非零值：实测 torch 2.11.0 会直接抛
> `RuntimeError: invalid low watermark ratio` 让 MPS 初始化失败；设 0.0 又与不设等价，故不设。
> `PYTORCH_MPS_FAST_MATH` / `PYTORCH_MPS_PREFER_METAL` 实测也零收益，均不加。

### ⚠️ 最大的性能陷阱：文本编码器默认在 CPU 上跑

`comfy/model_management.py` 里，MPS 会被强制设为 `vram_state = VRAMState.SHARED`，
而 `text_encoder_device()` 只在 `HIGH_VRAM / NORMAL_VRAM` 时才返回 GPU：

```python
def text_encoder_device():
    ...
    elif vram_state in (VRAMState.HIGH_VRAM, VRAMState.NORMAL_VRAM):
        return get_torch_device()
    else:
        return torch.device("cpu")      # ← MPS 的 SHARED 落到这里
```

**结论：不加 `--gpu-only` 时，8.7GB 的文本编码器整个在 CPU 上跑。** 脚本已默认加上。

### 速度归因（M4 Pro 实测，2026-10-16）

在 M4 Pro / 24GB 上实测（纯 MPS 基准，不加载模型）：

| 项目 | 实测值 | 结论 |
|---|---|---|
| MPS fp16 matmul | **5.8 TFLOPS**（各形状一致，硬上限） | 纯算力上限，是整档耗时的主因 |
| `torch.compile` | 13.44 ms vs eager 13.34 ms | 零收益，还要 5.5s 编译，**别开** |
| GGUF Q5_K 反量化 | 8 步共约 **4.3 秒**（占全程 <4%） | **换 Q4_K_M 收益可忽略**，不值得多下 4.2G |
| 注意力后端 | `sub_quad` 6.50 / `split` 7.60 / `SDPA` 8.26 ms | ComfyUI 默认的 `sub_quad` **已是三选一里最快**，无需改 |
| MPS 环境变量调优 | `FAST_MATH`/`PREFER_METAL` 均无变化 | 玄学，不加 |

**整档耗时主要由 MPS 算力决定，不是配置没调好。**（720 档整图实测：极速 8 步 117 秒 / 质量 20 步 459 秒 / 极速图生图 144 秒）
真要提速，有效手段只有两个（都会改变输出，需自己权衡）：

1. **降分辨率**——DiT 成本近似随 token 数线性增长，768 → 512 约快 2.25 倍（代价是出图变小）
2. `--gpu-only`（已默认开启）——省掉文本编码器在 CPU 上跑的那段

### MPS 已知限制

| 现象 | 原因 | 处理 |
|---|---|---|
| 出图能跑但明显慢 | int8 convrot / fp8 量化算子无 MPS 实现，回落 CPU | 换 bf16 权重（需内存 ≥36GB），或接受速度 |
| 报 OOM / 卡死不动 | 峰值超 24GB，触发 swap | 确认用的是 int8 文本编码器；关掉浏览器/其他大内存应用 |
| `Unknown model architecture!` | GGUF 补丁没打上 | 见 `patches/README.md` |
| 首个工作流加载节点报缺节点 | 改过 `custom_nodes/` 没重启 | 重启 ComfyUI |

## 目录结构

仓库只跟踪策划资产（根目录共 6 个文件 + 5 个目录），其余根目录文件属于 ComfyUI 运行时，见下一节：

```
├── workflows/                          # 7 个工作流（浏览器里直接拖进去打开）
├── styles/                             # 风格库：一个 txt = 一种风格，支持 # 注释
├── custom_nodes/comfyui-style-prompt/  # 自建节点：风格下拉选择器 + 存回节点 + 背景填充
├── patches/                            # ComfyUI-GGUF 的 Qwen-Image 架构补丁
├── scripts/
│   ├── prepare.py                      # 新机器一键准备（环境/运行时/依赖/节点补丁/模型）
│   ├── gen_ink_workflow.py             # 生成中国彩墨工作流 JSON
│   ├── make_t2i_fast.py                # 生成文生图-极速 JSON
│   ├── make_style_fast_only.py         # 精简图片风格转换为纯极速版
│   └── legacy/                         # 历史脚本，已不参与日常出图
├── docs/                               # 移植说明、性能剖析、模型配置笔记
└── start_comfyui.sh                    # 启动器（MPS 调优，自动设环境变量）
```

`start_comfyui.sh` 是 UTF-8 纯文本，直接用编辑器改即可，**没有编码坑**。
改完先干跑校验语法与探测逻辑，不会真的启动 ComfyUI：

```bash
bash -n start_comfyui.sh               # 语法检查
./start_comfyui.sh                     # 正常启动
```

它自己算路径、不写死用户名与环境（拷到任何目录都能跑），启动前会打印
Python 版本、torch 版本、**MPS 是否可用**、统一内存容量、以及本机/局域网访问地址。


## 根目录为什么还有 main.py / nodes.py

这个目录**同时是本机的 ComfyUI 运行环境**：`scripts/prepare.py` 会把官方 ComfyUI 浅克隆到
当前目录，补齐 `main.py`、`nodes.py`、`execution.py`、`server.py`、`comfy/`、`comfy_extras/`
等运行时文件，模型放 `models/`，出图进 `output/`。这些文件的位置由上游决定，**不能删也不能挪**。

`.gitignore` 用白名单模式（`/*` + 逐项放行），所以 `git status` 不会变成几十 GB 模型的灾难，
clone 出去也只拿到上面那些策划资产。`python scripts/prepare.py --check` 会列出缺什么。

## 免责声明

本项目只包含工作流与代码，不含任何模型权重与图片素材。模型的使用请遵守各自许可证；
生成内容（尤其是人物风格化）请遵守当地法律法规，不要用于伪造他人身份等用途。
