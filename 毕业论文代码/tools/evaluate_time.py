import os
import sys
import time
import torch
import numpy as np
import subprocess

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from rapnet_hevc_ppo_qp_yuv_simple import (
    RAPNetYCbCr, CONFIG,
    read_yuv420_clip_to_planes,
    downsample_yuv420_clip,
    ycbcr420_clip_to_yuv420_bytes
)

# =======================
# 1. 基本配置（只改这里）
# =======================
VIDEO_PATH = os.environ.get(
    "RAPNET_VIDEO_PATH",
    "/datasets/UVG/original_yuv/Bosphorus_1920x1080_120fps_420_8bit_YUV.yuv",
)
W, H = 1920, 1080
VIDEO_ID = "bosphorus"
QP_LIST = [int(q) for q in os.environ.get("RAPNET_QP_LIST", "37,42,45").split(",")]

CKPT = os.environ.get(
    "RAPNET_ACTOR_CKPT",
    os.path.join(ROOT_DIR, "checkpoints", "actor_best.pth"),
)
DEVICE = os.environ.get("RAPNET_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = os.environ.get("RAPNET_TMP_DIR", os.path.join(ROOT_DIR, "tmp"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =======================
# 2. 加载模型
# =======================
device = torch.device(DEVICE)
actor = RAPNetYCbCr().to(device).eval()
actor.load_state_dict(torch.load(CKPT, map_location=device))

# =======================
# 3. 读视频 & 下采样
# =======================
frame_bytes = int(W * H * 1.5)
total_frames = os.path.getsize(VIDEO_PATH) // frame_bytes
t0 = time.perf_counter()
y, cb, cr = read_yuv420_clip_to_planes(VIDEO_PATH, total_frames, H, W)
t_read = time.perf_counter() - t0

t0 = time.perf_counter()
y_ds, cb_ds, cr_ds = downsample_yuv420_clip(
    y, cb, cr, scale=CONFIG["downscale"]
)
t_ds = time.perf_counter() - t0

# =======================
# 4. RAPNet 前处理计时
# =======================
chunk = CONFIG["clip_len"]
y_out, cb_out, cr_out = [], [], []

t0 = time.perf_counter()

with torch.no_grad():
    for i in range(0, total_frames, chunk):
        end = min(i + chunk, total_frames)

        y_in  = torch.from_numpy(y_ds[i:end]).float().to(device).unsqueeze(0)
        cb_in = torch.from_numpy(cb_ds[i:end]).float().to(device).unsqueeze(0)
        cr_in = torch.from_numpy(cr_ds[i:end]).float().to(device).unsqueeze(0)

        y_a, cb_a, cr_a = actor(y_in, cb_in, cr_in)

        y_out.append(y_a.squeeze(0).cpu().numpy())
        cb_out.append(cb_a.squeeze(0).cpu().numpy())
        cr_out.append(cr_a.squeeze(0).cpu().numpy())

t_pre = time.perf_counter() - t0

y_out  = np.concatenate(y_out, axis=0)
cb_out = np.concatenate(cb_out, axis=0)
cr_out = np.concatenate(cr_out, axis=0)

# =======================
# 5. 写 yuv（一次即可）
# =======================
tmp_yuv = os.path.join(OUTPUT_DIR, "rapnet_processed.yuv")
with open(tmp_yuv, "wb") as f:
    f.write(ycbcr420_clip_to_yuv420_bytes(y_out, cb_out, cr_out))

H_ds, W_ds = y_out.shape[-2:]

# =======================
# 6. 编码计时（QP 序列）
# =======================
print(f"\nVideo: {VIDEO_ID}, Frames: {total_frames}")
print(f"Read: {t_read*1000/total_frames:.3f} ms/frame")
print(f"Downsample: {t_ds*1000/total_frames:.3f} ms/frame")
print(f"Preprocess: {t_pre*1000/total_frames:.3f} ms/frame\n")

for qp in QP_LIST:
    bitstream = os.path.join(OUTPUT_DIR, f"rapnet_qp{qp}_{VIDEO_ID}.hevc")

    # ---------- Encode ----------
    cmd_enc = [
        CONFIG["ffmpeg_bin"], "-y",
        "-f", "rawvideo",
        "-pix_fmt", "yuv420p",
        "-s:v", f"{W_ds}x{H_ds}",
        "-r", str(CONFIG["fps"]),
        "-i", tmp_yuv,
        "-c:v", "hevc_nvenc",
        "-rc", "constqp",
        "-qp", str(qp),
        "-preset", "p4",
        bitstream
    ]

    t0 = time.perf_counter()
    subprocess.run(cmd_enc, check=True)
    # subprocess.run(cmd_enc, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t_enc = time.perf_counter() - t0

    # ---------- Decode ----------
    # hevc -> raw yuv420p
    cmd_dec = [
        CONFIG["ffmpeg_bin"], "-y",
        "-c:v", "hevc_cuvid",
        "-i", bitstream,
        "-f", "rawvideo",
        "-pix_fmt", "yuv420p",
        "-"
    ]

    t0 = time.perf_counter()
    subprocess.run(cmd_dec, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t_dec = time.perf_counter() - t0

    # ---------- Report ----------
    ms_read  = t_read * 1000 / total_frames
    ms_ds    = t_ds * 1000 / total_frames
    ms_pre   = t_pre * 1000 / total_frames
    ms_enc   = t_enc * 1000 / total_frames
    ms_dec   = t_dec * 1000 / total_frames
    ms_total = ( t_read + t_ds + t_pre + t_enc + t_dec) * 1000 / total_frames

    print(
        f"QP {qp:2d} | "
        f" Read: {ms_read:.3f} |Downsample: {ms_ds:.3f} | Pre: {ms_pre:.3f} | Enc: {ms_enc:.3f} | Dec: {ms_dec:.3f} | "
        f"E2E: {ms_total:.3f} ms/frame"
    )
