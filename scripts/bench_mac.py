"""M4 Pro 本机基准：文生图 turbo 8 步 / 质量档 20 步 / 图生图。
只读项目文件，产物写 _perf/，不改动工作流。
"""
import json
import os
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8188"
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"
urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))

PROMPT = ("A serene Chinese ink-wash landscape with misty mountains and a quiet river, "
          "soft muted tones, cinematic lighting")
NEG = "overexposed, blown out highlights, washed out, low contrast"

TURBO_LORA = "qwen_image_2.1_viggle_turbo_v0.2.1_r128_comfy.safetensors"


def post(path, payload, timeout=120):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def get(path):
    req = urllib.request.Request(BASE + path)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def build(steps, cfg, lora, use_cache, resolution, ref_image, prompt):
    g = {
        "1": {"class_type": "UnetLoaderGGUF",
              "inputs": {"unet_name": "qwen-image-2.1-Q5_K_M.gguf"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": "qwen3vl_8b_int8_convrot.safetensors",
                         "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader",
              "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
        "5": {"class_type": "TextEncodeQwenImage21",
              "inputs": {"clip": ["2", 0], "prompt": prompt, "negative_prompt": NEG,
                         "vae": ["3", 0], "resolution": resolution}},
    }
    if ref_image:
        g["4"] = {"class_type": "LoadImage", "inputs": {"image": ref_image}}
        g["5"]["inputs"]["images.image_1"] = ["4", 0]
    src = ["1", 0]
    if lora:
        g["16"] = {"class_type": "LoraLoaderBypassModelOnly",
                   "inputs": {"model": ["1", 0], "lora_name": TURBO_LORA,
                              "strength_model": 1.0}}
        src = ["16", 0]
    if use_cache:
        g["9"] = {"class_type": "QwenImage21Cache",
                  "inputs": {"model": src, "device": "auto", "dtype": "default"}}
        src = ["9", 0]
    g["6"] = {"class_type": "KSampler",
              "inputs": {"model": src, "positive": ["5", 0], "negative": ["5", 1],
                         "latent_image": ["5", 2], "seed": 42, "steps": steps, "cfg": cfg,
                         "sampler_name": "euler", "scheduler": "beta", "denoise": 1.0}}
    g["7"] = {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}}
    g["8"] = {"class_type": "SaveImage",
              "inputs": {"images": ["7", 0], "filename_prefix": "macbench"}}
    return g


def run(label, steps, cfg, lora=False, use_cache=False, resolution=720,
        ref_image=None, prompt=PROMPT):
    g = build(steps, cfg, lora, use_cache, resolution, ref_image, prompt)
    t0 = time.perf_counter()
    try:
        res = post("/prompt", {"prompt": g, "client_id": "macbench"})
    except Exception as exc:
        print(f"{label:24s} 提交失败: {exc}")
        return None
    pid = res["prompt_id"]
    while True:
        h = get("/history/" + pid)
        if pid in h:
            break
        time.sleep(1.0)
    total = time.perf_counter() - t0
    st = h[pid].get("status", {})
    ok = st.get("status_str") == "success"
    print(f"{label:24s} total={total:6.1f}s  {'OK' if ok else 'FAIL'}", flush=True)
    if not ok:
        for m in st.get("messages", []):
            if m[0] == "execution_error":
                print("   ERROR:", json.dumps(m[1])[:300])
    return total


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "all"
    results = {}
    if p in ("all", "cold"):
        # 冷启动：含模型首次加载（GGUF 反量化 + 上 GPU）
        results["turbo8-720-冷启动"] = run(
            "turbo8-720-冷启动", 8, 1.0, lora=True, use_cache=True,
            prompt=PROMPT + " cold")
    if p in ("all", "turbo"):
        results["turbo8-720-热"] = run(
            "turbo8-720-热", 8, 1.0, lora=True, use_cache=True,
            prompt=PROMPT + " warm1")
    if p in ("all", "quality"):
        results["quality20-720-热"] = run(
            "quality20-720-热", 20, 3.5, use_cache=True,
            prompt=PROMPT + " warm2")
    if p in ("all", "img2img"):
        results["turbo8-图生图-热"] = run(
            "turbo8-图生图-热", 8, 1.0, lora=True, use_cache=True,
            ref_image="1.jpg", prompt=PROMPT + " warm3")
    print("\n=== 汇总 ===")
    for k, v in results.items():
        if v is not None:
            print(f"  {k:22s} {v:6.1f} 秒")
