import os
import json
import math
import random
import tempfile
import subprocess
from typing import Dict, List, Tuple
import re
import glob
import shutil
import logging
import cv2

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lpips  # LPIPS 感知损失库
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

from gradient_monitor import diagnose_ac_training, grad_stats

SUBMISSION_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_TMP_DIR = os.environ.get("RAPNET_TMP_DIR", os.path.join(SUBMISSION_ROOT, "tmp"))
DEFAULT_RUN_DIR = os.environ.get("RAPNET_RUN_DIR", os.path.join(SUBMISSION_ROOT, "runs"))
DEFAULT_RESULT_DIR = os.environ.get("RAPNET_RESULT_DIR", os.path.join(SUBMISSION_ROOT, "results"))
DEFAULT_CKPT_DIR = os.environ.get("RAPNET_CHECKPOINT_DIR", os.path.join(SUBMISSION_ROOT, "checkpoints"))

# ================================================================
# 基础设置
# ================================================================


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    if torch.cuda.device_count() > 2:
        LPIPS_DEVICE = torch.device("cuda:1")
    else:
        LPIPS_DEVICE = torch.device("cuda:1")
else:
    LPIPS_DEVICE = torch.device("cpu")
# 分开给 Actor / Critic 用的设备
DEVICE_ACTOR = torch.device("cuda:1") if torch.cuda.is_available() else device
DEVICE_CRITIC = torch.device("cuda:2") if torch.cuda.device_count() > 1 else DEVICE_ACTOR
CONFIG = {
    
    # ---------- 硬件与设备 ----------
    # 建议: Actor 放在 GPU 0，Critic 放在 GPU 1 (如果可用)
    # 如果只有 1 个 GPU，代码会自动检测并全部回退到 cuda:0
    "device_actor": "cuda:1",   
    "device_critic": "cuda:2",
    
    # ---------- 迭代 & 数据 ----------
    "num_iterations": 10000,    # 论文里就是 5000 iterations
    "clip_len": 16,            # 1080p 一次最多处理 16 帧（论文中对 1080p 的限制）
    "width": 1920,
    "height": 1080,
    "downscale": 2,

    # ---------- 学习率 ----------
    "lr_actor": 1e-5,
    "lr_critic": 1e-5,
    # 学习率从 1e-4 开始，每隔 1000 iterations 下降为原来的 1/3
    "lr_step_size": 1000,
    "lr_gamma": 1.0 / 3.0,

    # ---------- RD & env ----------
    "qp_list": [37, 40, 42, 45, 47],
  # 核心修改：将奖励权重提高
    "alpha_q": 8,    
    "beta_r":2,     
    "fps": 120,                # Beauty 原始是 120fps，你可以用 120，或者统一成 120 也行，关键是 baseline 和训练一致
    "ffmpeg_bin": "ffmpeg",
    "work_dir": os.path.join(DEFAULT_TMP_DIR, "rl_nvenc_tmp"),

    # ---------- eval / save / log ----------
    "eval_interval": 100,      # 每 50 iter 做一次验证
    "eval_num_clips": 16,       # 每次验证用 16 个 clip 求平均 reward
    "save_interval": 100,      # 每 100 iter 存一次 checkpoint
    "log_interval": 10,        # 每 10 iter 打印一条 log
    "log_dir": os.path.join(DEFAULT_RUN_DIR, "rapnet_hevc_1_22_v40"),
    "model_dir": os.path.join(DEFAULT_CKPT_DIR, "rapnet_hevc_1_22_v40"),
    "grad_check_interval" :  10, #梯度检查间隔
    "image_interval" : 50   # 你想多久可视化一次图就改这里
}

# ================================================================
# 精确断点续训工具函数
# ================================================================
def get_last_iter_simple(log_dir: str, model_dir: str) -> int:
    """
    从checkpoint文件名推断最后的iteration号（最快最可靠）
    
    原理：
    - 查找model_dir中最大iter号的checkpoint文件
    - 从文件名 actor_iter000123.pth 提取iter号
    
    返回：
    - 最后记录的iteration号（从0开始如果没有checkpoint）
    
    使用：
    >>> last_iter = get_last_iter_simple(log_dir, model_dir)
    >>> print(f"最后训练到iter {last_iter}，将从iter {last_iter+1}继续")
    """
    import os
    import glob
    
    if not os.path.exists(model_dir):
        print(f"[Info] Model目录不存在: {model_dir}，从iter 1开始")
        return 0
    
    # 找到所有iter checkpoint
    iter_files = glob.glob(os.path.join(model_dir, "actor_iter*.pth"))
    
    if not iter_files:
        print(f"[Info] 未找到iter checkpoint，从iter 1开始")
        return 0
    
    # 从文件名提取iter号，找最大的
    max_iter = 0
    for fpath in iter_files:
        try:
            fname = os.path.basename(fpath)
            # 格式: actor_iter000123.pth
            iter_str = fname.split("iter")[1].split(".")[0]
            iter_num = int(iter_str)
            max_iter = max(max_iter, iter_num)
        except:
            pass
    
    if max_iter > 0:
        print(f"[Info] 找到最后的checkpoint: iter {max_iter}，将从iter {max_iter + 1}继续")
    
    return max_iter



# ================================================================
DEBUG_SHAPES = False   # 打印维度和数值范围

# 全局 LPIPS 模型实例（避免每次调用都重新构建）
_lpips_model = None

def resolve_devices():
    """根据 CONFIG 解析 Actor / Critic 使用的 device。"""
    if not torch.cuda.is_available():
        print("[Warn] CUDA 不可用，Actor/Critic 都回退到 CPU。")
        return torch.device("cpu"), torch.device("cpu")

    n_gpu = torch.cuda.device_count()
    dev_actor = torch.device(CONFIG["device_actor"])

    if n_gpu >= 2:
        dev_critic = torch.device(CONFIG["device_critic"])
    else:
        print(f"[Warn] 只有 {n_gpu} 张 GPU，Actor/Critic 都使用 {dev_actor}。")
        dev_critic = dev_actor

    print(f"[*] Actor device = {dev_actor}, Critic device = {dev_critic}")
    return dev_actor, dev_critic

def init_lpips_model(net: str = "alex"): # 将默认值改为 alex
    global _lpips_model
    if _lpips_model is None:
        print(f"[*] Initializing LPIPS on {LPIPS_DEVICE}...")
        # 确保使用 alex，这与 stage_3 的 lpips.LPIPS(net='alex') 一致
        model = lpips.LPIPS(net='alex').to(LPIPS_DEVICE) 
        model.eval()
        _lpips_model = model
    return _lpips_model

# ================================================================
# 数据读取 / 预处理工具
# ================================================================
def load_clip_frames(meta: Dict, clip_len: int) -> np.ndarray:
    """
    针对 .yuv 文件的加载函数（旧版 BGR 路径，当前训练逻辑不再依赖）
    meta 需包含: {'path': 'xxx.yuv', 'width': 1920, 'height': 1080}
    """
    path = meta["path"]
    width = meta["width"]
    height = meta["height"]

    loader = YUVLoader(path, width, height)

    # 随机采样起始帧
    if loader.total_frames <= clip_len:
        start_idx = 0
    else:
        start_idx = random.randint(0, loader.total_frames - clip_len)

    return loader.read_clip(start_idx, clip_len)


class YUVLoader:
    """
    专门用于读取 raw YUV420p 文件的加载器（旧版：返回 BGR，用于兼容）。
    相比 cv2.VideoCapture，它支持无损读取和快速随机访问 (seek)。
    """
    def __init__(self, file_path, width, height, pixel_format="yuv420p"):
        self.path = file_path
        self.w = width
        self.h = height
        # 计算单帧字节数
        if pixel_format == "yuv420p":
            self.frame_bytes = int(width * height * 1.5)
        else:
        # ...
            raise NotImplementedError("Only yuv420p is supported")

        self.file_size = os.path.getsize(file_path)
        self.total_frames = self.file_size // self.frame_bytes

    def read_clip(self, start_frame, clip_len):
        """
        读取从 start_frame 开始的 clip_len 帧，并返回 BGR clip。
        """
        import cv2

        if start_frame + clip_len > self.total_frames:
            raise ValueError("Requested frames exceed total frames")

        frames_bgr = []
        with open(self.path, "rb") as f:
            # 1. 跳转到起始帧
            f.seek(start_frame * self.frame_bytes)

            for _ in range(clip_len):
                # 2. 读取一帧的二进制数据
                raw = f.read(self.frame_bytes)
                if len(raw) < self.frame_bytes:
                    break

                # 3. 解析 YUV
                # Y 分量大小: W * H
                y_size = self.w * self.h
                # UV 分量大小: (W/2) * (H/2)
                uv_size = (self.w // 2) * (self.h // 2)

                Y = np.frombuffer(raw[0:y_size], dtype=np.uint8).reshape(self.h, self.w)
                U = np.frombuffer(raw[y_size:y_size + uv_size], dtype=np.uint8).reshape(self.h // 2, self.w // 2)
                V = np.frombuffer(raw[y_size + uv_size:], dtype=np.uint8).reshape(self.h // 2, self.w // 2)

                # 上采样到全分辨率
                U_up = cv2.resize(U, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
                V_up = cv2.resize(V, (self.w, self.h), interpolation=cv2.INTER_LINEAR)

                # 合并为 YUV444（这里假定顺序 Y, U, V）
                yuv = cv2.merge([Y, U_up, V_up])
                # 转换到 BGR
                bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
                frames_bgr.append(bgr)

        if len(frames_bgr) < clip_len:
            raise RuntimeError(f"Not enough frames in {self.path}")

        return np.stack(frames_bgr, axis=0)  # (T, H, W, 3)


def lanczos_resize(frame: np.ndarray, dst_w: int, dst_h: int) -> np.ndarray:
    import cv2
    return cv2.resize(frame, (dst_w, dst_h), interpolation=cv2.INTER_LANCZOS4)


def bgr_clip_to_ycbcr420(frames_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    BGR clip -> Y/Cb/Cr 4:2:0, 归一化到 0~1：
        Y:  (T,1,H,W)
        Cb: (T,1,H/2,W/2)
        Cr: (T,1,H/2,W/2)
    """
    import cv2

    T, H, W, _ = frames_bgr.shape
    y_list = []
    cb_list = []
    cr_list = []

    for t in range(T):
        bgr = frames_bgr[t]
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        Y = ycrcb[:, :, 0]
        Cr = ycrcb[:, :, 1]
        Cb = ycrcb[:, :, 2]

        Cb_ds = cv2.resize(Cb, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        Cr_ds = cv2.resize(Cr, (W // 2, H // 2), interpolation=cv2.INTER_AREA)

        y_list.append(Y.astype(np.float32) / 255.0)
        cb_list.append(Cb_ds.astype(np.float32) / 255.0)
        cr_list.append(Cr_ds.astype(np.float32) / 255.0)

    y = np.stack(y_list, axis=0)[:, None, :, :]
    cb = np.stack(cb_list, axis=0)[:, None, :, :]
    cr = np.stack(cr_list, axis=0)[:, None, :, :]
    return y, cb, cr


def ycbcr420_clip_to_yuv420_bytes(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> bytes:
    """
    将 Y/Cb/Cr 4:2:0 (float32, 0~1) 序列打包成连续的 yuv420p 字节流。
    """
    T, _, H, W = y.shape
    Hc, Wc = cb.shape[2], cb.shape[3]
    assert Hc * 2 == H and Wc * 2 == W

    frames_bytes = []
    for t in range(T):
        Y = np.clip(y[t, 0] * 255.0, 0, 255).astype(np.uint8)
        Cb = np.clip(cb[t, 0] * 255.0, 0, 255).astype(np.uint8)
        Cr = np.clip(cr[t, 0] * 255.0, 0, 255).astype(np.uint8)

        frames_bytes.append(Y.tobytes())
        frames_bytes.append(Cb.tobytes())
        frames_bytes.append(Cr.tobytes())

    return b"".join(frames_bytes)


def read_yuv420_clip_to_bgr(path: str, T: int, H: int, W: int) -> np.ndarray:
    """
    从 yuv420p 文件中读出 T 帧，返回 (T,H,W,3) BGR。
    """
    import cv2

    frame_size = H * W * 3 // 2
    with open(path, "rb") as f:
        data = f.read()

    expected = frame_size * T
    if len(data) < expected:
        raise RuntimeError("yuv file is too short")

    frames = []
    offset = 0
    for _ in range(T):
        frame_bytes = data[offset:offset + frame_size]
        offset += frame_size

        y_size = H * W
        c_size = (H // 2) * (W // 2)

        Y = np.frombuffer(frame_bytes[0:y_size], dtype=np.uint8).reshape(H, W)
        Cb = np.frombuffer(frame_bytes[y_size:y_size + c_size], dtype=np.uint8).reshape(H // 2, W // 2)
        Cr = np.frombuffer(frame_bytes[y_size + c_size:y_size + 2 * c_size], dtype=np.uint8).reshape(H // 2, W // 2)

        Cb_up = cv2.resize(Cb, (W, H), interpolation=cv2.INTER_LINEAR)
        Cr_up = cv2.resize(Cr, (W, H), interpolation=cv2.INTER_LINEAR)

        ycrcb = np.stack([Y, Cr_up, Cb_up], axis=2)
        bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        frames.append(bgr)

    return np.stack(frames, axis=0)


def read_yuv420_clip_to_planes(path: str, T: int, H: int, W: int):
    """从 yuv420p 文件中读出 T 帧，返回归一化到 0~1 的 Y/Cb/Cr 平面。

    返回:
        y:  (T,1,H,W)
        cb: (T,1,H/2,W/2)
        cr: (T,1,H/2,W/2)
    """
    frame_size = H * W * 3 // 2
    y_list, cb_list, cr_list = [], [], []
    with open(path, "rb") as f:
        for _ in range(T):
            frame_bytes = f.read(frame_size)
            if len(frame_bytes) < frame_size:
                raise RuntimeError("yuv file is too short")

            y_size = H * W
            c_size = (H // 2) * (W // 2)

            Y = np.frombuffer(frame_bytes[0:y_size], dtype=np.uint8).reshape(H, W)
            Cb = np.frombuffer(frame_bytes[y_size:y_size + c_size], dtype=np.uint8).reshape(H // 2, W // 2)
            Cr = np.frombuffer(frame_bytes[y_size + c_size:y_size + 2 * c_size], dtype=np.uint8).reshape(H // 2, W // 2)

            y_list.append(Y.astype(np.float32) / 255.0)
            cb_list.append(Cb.astype(np.float32) / 255.0)
            cr_list.append(Cr.astype(np.float32) / 255.0)

    y = np.stack(y_list, axis=0)[:, None, :, :]
    cb = np.stack(cb_list, axis=0)[:, None, :, :]
    cr = np.stack(cr_list, axis=0)[:, None, :, :]
    return y, cb, cr

# --- action stats (Y/Cb/Cr) ---
def _stats(a, s):
    a = a.astype(np.float32)
    s = s.astype(np.float32)
    d = a - s
    mean = float(a.mean())
    std = float(a.std())
    delta_mean = float(np.abs(d).mean())
    delta_l2 = float(np.sqrt(np.mean(d * d)))
    sat0 = float(np.mean(a < 1e-3))
    sat1 = float(np.mean(a > 1.0 - 1e-3))
    return mean, std, delta_mean, delta_l2, sat0, sat1

#---计算Actor 改动的“高频能量” / 改动的“总能量”
def high_freq_ratio(delta_y):
    # delta_y: (BT,1,H,W) or (B,T,1,H,W)
    if delta_y.dim() == 5:
        B, T, C, H, W = delta_y.shape
        delta_y = delta_y.view(B*T, C, H, W)

    lap = torch.tensor([[0,-1,0],[-1,4,-1],[0,-1,0]],
                       device=delta_y.device,
                       dtype=delta_y.dtype).view(1,1,3,3)

    hf = F.conv2d(delta_y, lap, padding=1)
    return torch.mean(hf**2) / (torch.mean(delta_y**2) + 1e-8)

def blur5x5(x):
    k = torch.ones((1,1,5,5), device=x.device, dtype=x.dtype) / 25.0
    x = F.pad(x, (2,2,2,2), mode="reflect")
    return F.conv2d(x, k)

def high_freq_ratio_01(delta):
    if delta.dim() == 5:
        B,T,C,H,W = delta.shape
        delta = delta.view(B*T, C, H, W)

    low = blur5x5(delta)
    high = delta - low
    return torch.mean(high**2) / (torch.mean(delta**2) + 1e-8)
#------EdgeWeightedDelta / r_edge ——「是否在边缘上动刀」
def edge_weighted_delta(y, y_hat):
    # y, y_hat: (BT,1,H,W) or (B,T,1,H,W)
    if y.dim() == 5:
        B, T, C, H, W = y.shape
        y = y.view(B*T, C, H, W)
        y_hat = y_hat.view(B*T, C, H, W)

    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                           device=y.device, dtype=y.dtype).view(1,1,3,3)
    sobel_y = sobel_x.transpose(2,3)

    gx = F.conv2d(y, sobel_x, padding=1)
    gy = F.conv2d(y, sobel_y, padding=1)
    edge = torch.sqrt(gx*gx + gy*gy + 1e-12)

    delta = torch.abs(y_hat - y)
    return torch.sum(delta * edge) / (torch.sum(delta) + 1e-8)
def _to_torch_btchw(x_np, device):
    """
    x_np: numpy array, shape (T,1,H,W) or (B,T,1,H,W)
    return: torch tensor float32 on device, shape (B*T,1,H,W) 方便 conv2d
    """
    x = torch.from_numpy(x_np).float()
    if x.dim() == 4:  # (T,1,H,W)
        x = x.unsqueeze(0)  # -> (1,T,1,H,W)
    # (B,T,1,H,W) -> (B*T,1,H,W)
    B, T, C, H, W = x.shape
    return x.to(device).view(B*T, C, H, W)

# ---------------------------------------------------------
# 直接在 YUV420 平面上操作的工具函数
# ---------------------------------------------------------

def load_clip_yuv_planes(meta: Dict, clip_len: int):
    """从 .yuv 文件直接读取 Y/Cb/Cr 平面 (float32, 0~1)。

    要求 meta 至少包含:
        {
            "id": "video_xxx",
            "path": "/path/to/xxx.yuv",
            "width": 1920,
            "height": 1080,
        }
    返回:
        y:  (T,1,H,W)
        cb: (T,1,H/2,W/2)
        cr: (T,1,H/2,W/2)
    """
    path = meta["path"]
    W = int(meta["width"])
    H = int(meta["height"])

    frame_bytes = int(W * H * 3 / 2)
    file_size = os.path.getsize(path)
    total_frames = file_size // frame_bytes
    if total_frames <= clip_len:
        start_idx = 0
    else:
        start_idx = random.randint(0, total_frames - clip_len)

    y_list, cb_list, cr_list = [], [], []
    with open(path, "rb") as f:
        f.seek(start_idx * frame_bytes)
        for _ in range(clip_len):
            raw = f.read(frame_bytes)
            if len(raw) < frame_bytes:
                raise RuntimeError(f"Not enough frames in {path}")

            y_size = W * H
            uv_size = (W // 2) * (H // 2)

            Y = np.frombuffer(raw[0:y_size], dtype=np.uint8).reshape(H, W)
            U = np.frombuffer(raw[y_size:y_size + uv_size], dtype=np.uint8).reshape(H // 2, W // 2)
            V = np.frombuffer(raw[y_size + uv_size:y_size + 2 * uv_size], dtype=np.uint8).reshape(H // 2, W // 2)

            y_list.append(Y.astype(np.float32) / 255.0)
            cb_list.append(U.astype(np.float32) / 255.0)
            cr_list.append(V.astype(np.float32) / 255.0)

    y = np.stack(y_list, axis=0)[:, None, :, :]   # (T,1,H,W)
    cb = np.stack(cb_list, axis=0)[:, None, :, :] # (T,1,H/2,W/2)
    cr = np.stack(cr_list, axis=0)[:, None, :, :] # (T,1,H/2,W/2)
    return y, cb, cr


def downsample_yuv420_clip(y, cb, cr, scale: int = 2):
    """对 YUV420 clip 做空域下采样。

    输入:
        y:  (T,1,H,W)
        cb: (T,1,H/2,W/2)
        cr: (T,1,H/2,W/2)
    输出:
        y_ds:  (T,1,H/scale,   W/scale)
        cb_ds: (T,1,H/(2*scale), W/(2*scale))
        cr_ds: 同上
    """
    import cv2

    T, _, H, W = y.shape
    Hc, Wc = cb.shape[2], cb.shape[3]
    assert Hc * 2 == H and Wc * 2 == W

    H_ds, W_ds = H // scale, W // scale
    Hc_ds, Wc_ds = Hc // scale, Wc // scale

    y_ds = np.zeros((T, 1, H_ds, W_ds), dtype=np.float32)
    cb_ds = np.zeros((T, 1, Hc_ds, Wc_ds), dtype=np.float32)
    cr_ds = np.zeros((T, 1, Hc_ds, Wc_ds), dtype=np.float32)

    for t in range(T):
        y_ds[t, 0] = cv2.resize(y[t, 0], (W_ds, H_ds), interpolation=cv2.INTER_LANCZOS4)
        cb_ds[t, 0] = cv2.resize(cb[t, 0], (Wc_ds, Hc_ds), interpolation=cv2.INTER_AREA)
        cr_ds[t, 0] = cv2.resize(cr[t, 0], (Wc_ds, Hc_ds), interpolation=cv2.INTER_AREA)

    return y_ds, cb_ds, cr_ds


def upsample_yuv420_clip(y, cb, cr, scale: int = 2):
    """对 YUV420 clip 做空域上采样。

    输入:
        y:  (T,1,H,W)
        cb: (T,1,H/2,W/2)
        cr: (T,1,H/2,W/2)
    输出:
        y_up:  (T,1,H*scale,W*scale)
        cb_up: (T,1,(H/2)*scale,(W/2)*scale)
        cr_up: 同上
    """
    import cv2

    T, _, H, W = y.shape
    Hc, Wc = cb.shape[2], cb.shape[3]
    assert Hc * 2 == H and Wc * 2 == W

    H_up, W_up = H * scale, W * scale
    Hc_up, Wc_up = Hc * scale, Wc * scale

    y_up = np.zeros((T, 1, H_up, W_up), dtype=np.float32)
    cb_up = np.zeros((T, 1, Hc_up, Wc_up), dtype=np.float32)
    cr_up = np.zeros((T, 1, Hc_up, Wc_up), dtype=np.float32)

    for t in range(T):
        y_up[t, 0] = cv2.resize(y[t, 0], (W_up, H_up), interpolation=cv2.INTER_LANCZOS4)
        cb_up[t, 0] = cv2.resize(cb[t, 0], (Wc_up, Hc_up), interpolation=cv2.INTER_AREA)
        cr_up[t, 0] = cv2.resize(cr[t, 0], (Wc_up, Hc_up), interpolation=cv2.INTER_AREA)

    return y_up, cb_up, cr_up


def yuv420_planes_to_bgr_clip(y, cb, cr):
    """将 Y/Cb/Cr 4:2:0 clip 转为 BGR (uint8)，用于 LPIPS / 可视化。"""
    import cv2

    T, _, H, W = y.shape
    Hc, Wc = cb.shape[2], cb.shape[3]
    assert Hc * 2 == H and Wc * 2 == W

    y_u8 = np.clip(y * 255.0, 0, 255).astype(np.uint8)
    cb_u8 = np.clip(cb * 255.0, 0, 255).astype(np.uint8)
    cr_u8 = np.clip(cr * 255.0, 0, 255).astype(np.uint8)

    frames = []
    for t in range(T):
        Y = y_u8[t, 0]
        Cb = cb_u8[t, 0]
        Cr = cr_u8[t, 0]

        Cb_up = cv2.resize(Cb, (W, H), interpolation=cv2.INTER_LINEAR)
        Cr_up = cv2.resize(Cr, (W, H), interpolation=cv2.INTER_LINEAR)

        ycrcb = np.stack([Y, Cr_up, Cb_up], axis=2)
        bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        frames.append(bgr)

    return np.stack(frames, axis=0)  # (T,H,W,3)


def compute_quality_metric(ref_clip_1080p: np.ndarray,
                           dec_clip_1080p: np.ndarray) -> float:
    """
    使用 PyTorch 版 LPIPS 作为质量指标。
    修正版：逐帧计算以节省显存 (Batch Size = 1)。
    输入:
        ref_clip_1080p: (T, H, W, 3) uint8, BGR
        dec_clip_1080p: (T, H, W, 3) uint8, BGR
    """
    # 1) 初始化 LPIPS 模型
    loss_fn = init_lpips_model(net="alex")
    
    T = ref_clip_1080p.shape[0]
    scores = []

    # 使用 no_grad 上下文，避免计算图占用显存
    with torch.no_grad():
        for t in range(T):
            # 2) 逐帧提取 & 预处理
            # 取出一帧 (H, W, 3) BGR
            ref_frame = ref_clip_1080p[t]
            dec_frame = dec_clip_1080p[t]
            
            # BGR -> RGB (利用切片反转) -> Tensor (1, 3, H, W)
            # copy() 是为了确保内存连续，避免 PyTorch 警告
            # permute(2, 0, 1) 将 (H, W, C) 转为 (C, H, W)
            # unsqueeze(0) 增加 Batch 维度 -> (1, C, H, W)
            ref_tensor = torch.from_numpy(ref_frame[..., ::-1].copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            dec_tensor = torch.from_numpy(dec_frame[..., ::-1].copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            
            # 归一化到 [-1, 1] (LPIPS 要求)
            ref_tensor = ref_tensor * 2.0 - 1.0
            dec_tensor = dec_tensor * 2.0 - 1.0
            
            # 移动到 GPU
            ref_tensor = ref_tensor.to(LPIPS_DEVICE)
            dec_tensor = dec_tensor.to(LPIPS_DEVICE)
            
            # 3) 计算单帧 LPIPS
            dist = loss_fn(ref_tensor, dec_tensor)
            scores.append(dist.item())
            
            # (可选) 显式删除变量，加速显存回收
            del ref_tensor, dec_tensor, dist

    # 4) 计算平均分并取负 (因为 LPIPS 越小越好，Reward 越大越好)
    avg_lpips = np.mean(scores)
    quality = -float(avg_lpips)
    
    return quality

# ================================================================
# 辅助函数：日志解析与帧质量计算
# ================================================================

def parse_ffmpeg_log_for_video_size(log_path: str) -> float:
    """
    从 FFmpeg stderr 日志中提取视频流大小（kB），排除容器头信息。
    寻找类似 "video:208kB audio:0kB ..." 的行。
    """
    video_size_kb = 0.0
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
        # 匹配最后统计行的 video:xxxxkB
        # 典型输出: video:15kB audio:0kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 12.500%
        match = re.search(r"video:([0-9\.]+)kB", content)
        if match:
            video_size_kb = float(match.group(1))
        else:
            # 如果没找到统计行，回退到查找最后的 bitrate=xxx kbits/s (不推荐，含overhead)
            # 但对于 NVENC constqp，通常都有统计行
            print(f"[Warn] Could not parse video stream size from {log_path}, logging output may be truncated.")
            
    return video_size_kb

# ================================================================
# 辅助函数区域 (在 Class 定义之外)
# ================================================================

def sobel_edge_mag(x4):  # x4: (N,1,H,W) or (N,C,H,W) we'll reduce to 1ch
    if x4.size(1) > 1:
        x = x4.mean(dim=1, keepdim=True)
    else:
        x = x4

    kx = torch.tensor([[-1., 0., 1.],
                       [-2., 0., 2.],
                       [-1., 0., 1.]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    ky = torch.tensor([[-1., -2., -1.],
                       [ 0.,  0.,  0.],
                       [ 1.,  2.,  1.]], device=x.device, dtype=x.dtype).view(1,1,3,3)

    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-8)  # (N,1,H,W)
    return mag

def total_variation_loss_masked(img, weight=1.0, edge_k=20.0, edge_floor=0.05):
    """
    img: (B,T,C,H,W) or (N,C,H,W)
    edge_k: 越大越“保护边缘”（边缘处TV权重越小）
    edge_floor: 非边缘mask下限，防止mask全为0导致无梯度（建议 0.02~0.1）
    """
    if img.dim() == 5:
        b, t, c, h, w = img.size()
        x = img.view(-1, c, h, w)
    else:
        x = img

    n, c, h, w = x.size()

    # 1) edge magnitude on luminance-ish
    edge = sobel_edge_mag(x)  # (n,1,h,w)

    # 2) non-edge weight in [edge_floor, 1]
    non_edge = torch.exp(-edge_k * edge)
    non_edge = non_edge.clamp(min=edge_floor, max=1.0)  # (n,1,h,w)

    # 3) build masks aligned with TV diffs:
    # for vertical diffs: (x[:, :, 1:, :] - x[:, :, :-1, :]) shape (n,c,h-1,w)
    mh = non_edge[:, :, 1:, :] * non_edge[:, :, :-1, :]   # (n,1,h-1,w)
    # for horizontal diffs: shape (n,c,h,w-1)
    mw = non_edge[:, :, :, 1:] * non_edge[:, :, :, :-1]   # (n,1,h,w-1)

    dh = x[:, :, 1:, :] - x[:, :, :-1, :]
    dw = x[:, :, :, 1:] - x[:, :, :, :-1]

    # broadcast mask from 1ch to c
    tv_h = (dh * dh * mh).sum()
    tv_w = (dw * dw * mw).sum()

    return weight * (tv_h + tv_w) / (n * c * h * w)

def total_variation_loss(img, weight=1.0):
    # 如果输入是 5维 (B, T, C, H, W)，则合并前两维变成 (B*T, C, H, W)
    if img.dim() == 5:
        b, t, c, h, w = img.size()
        img = img.view(-1, c, h, w)
    
    # 现在 img 肯定是 4维的，可以正常拆包
    bs_img, c_img, h_img, w_img = img.size()
    
    tv_h = torch.pow(img[:,:,1:,:] - img[:,:,:-1,:], 2).sum()
    tv_w = torch.pow(img[:,:,:,1:] - img[:,:,:,:-1], 2).sum()
    
    return weight * (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)



# RAPNet / ValueNet
# ================================================================
class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        identity = x
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return identity + out


class RAPNetYCbCr(nn.Module):
    """
    RAPNet (Actor) 版本，加入 Pixel Unshuffle / Pixel Shuffle，

    输入:
        y:  (B,T,1,H,W)
        cb: (B,T,1,H/2,W/2)
        cr: (B,T,1,H/2,W/2)
    输出:
        y_out, cb_out, cr_out: 同形状
    """
    def __init__(self,
                 num_blocks: int = 5,
                 base_channels: int = 16,
                 scale: int = 2):
        super().__init__()
        self.scale = scale

        # Cb/Cr 上采样 / 下采样
        self.upsample_chroma = nn.Upsample(
            scale_factor=scale, mode="bilinear", align_corners=False
        )
        self.downsample_chroma = nn.Upsample(
            scale_factor=1.0 / scale, mode="bilinear", align_corners=False
        )

        # Pixel Unshuffle / Pixel Shuffle
        self.pixel_unshuffle = nn.PixelUnshuffle(scale)
        self.pixel_shuffle = nn.PixelShuffle(scale)

        # PixelUnshuffle 后通道数: 3 * scale^2
        in_channels = 3 * (scale ** 2)
        out_channels = 3 * (scale ** 2)

        self.head = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        self.body = nn.Sequential(*[ResBlock(base_channels) for _ in range(num_blocks)])
        self.tail = nn.Conv2d(base_channels, out_channels, 3, 1, 1)

        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, y, cb, cr):
        
        assert y.dim() == 5 and cb.dim() == 5 and cr.dim() == 5, \
            f"RAPNet expects (B,T,1,H,W), got {y.shape}, {cb.shape}, {cr.shape}"

        if DEBUG_SHAPES:
            print("[RAPNet] input shapes:",
                "y", y.shape, "cb", cb.shape, "cr", cr.shape)
            print("         input range y:",
                float(y.min()), float(y.max())) 
                   
        B, T, _, H, W = y.shape
        _, _, _, Hc, Wc = cb.shape
        assert Hc * self.scale == H and Wc * self.scale == W

        # 展平成 (BT, C, H, W) 便于卷积
        y_bt = y.view(B * T, 1, H, W)
        cb_bt = cb.view(B * T, 1, Hc, Wc)
        cr_bt = cr.view(B * T, 1, Hc, Wc)

        # Cb/Cr 上采样到亮度分辨率
        cb_up = self.upsample_chroma(cb_bt)  # (BT,1,H,W)
        cr_up = self.upsample_chroma(cr_bt)  # (BT,1,H,W)

        # 拼接 Y/Cb/Cr 得到输入 3 通道
        x_in = torch.cat([y_bt, cb_up, cr_up], dim=1)  # (BT,3,H,W)

        # Pixel Unshuffle: (BT,3,H,W) -> (BT, 3*scale^2, H/scale, W/scale)
        x = self.pixel_unshuffle(x_in)  # (BT, 3*4, H/2, W/2)

        # 主干网络（低分辨率上卷积+残差块）
        feat = self.act(self.head(x))
        feat = self.body(feat)
        feat = self.tail(feat)  # (BT, 3*4, H/2, W/2)

        # Pixel Shuffle: 恢复到原分辨率 3 通道
        delta = self.pixel_shuffle(feat)  # (BT,3,H,W)

        # 残差形式输出
        # 核心修改：使用 Tanh 将残差限制在 [-0.1, 0.1] 或 [-0.2, 0.2] 范围内。
        # 这意味着 Actor 最多只能改变像素值的 10%-20%，绝对不可能把 Cb 从 0.5 拉到 0.0。
        # 这样既防止了饱和(sat0=1)，也保证了梯度永远可以通过 Tanh 回传。
        # 1. 使用 Tanh 将残差限制在 [-0.1, 0.1] 范围内。
        # 让梯度在小范围内更连续，同时物理上限限制动作幅度.
        
        # 2. 【核心修改】方案 A：零均值约束
        # 对 B, C, H, W 的每个样本、每个通道，减去该通道的空间均值
        # keepdim=True 保证形状兼容 (B, C, 1, 1)
        delta = torch.tanh(delta) * 0.05 # 限制在 +/- 5% (约 12 个像素值)
        
        # 拆分 Y, Cb, Cr,Y可以改动均值，Cb，Cr不可以
        d_y = delta[:, 0:1, :, :]
        d_cb = delta[:, 1:2, :, :]
        d_cr = delta[:, 2:3, :, :]
        d_cb = d_cb - d_cb.mean(dim=(2, 3), keepdim=True)
        d_cr = d_cr - d_cr.mean(dim=(2, 3), keepdim=True)
        delta = torch.cat([d_y, d_cb, d_cr], dim=1)
        # delta = delta - delta.mean(dim=(2, 3), keepdim=True) 
        # delta = torch.clamp(delta, -0.05, 0.05)  
        
        out_3 = torch.clamp(x_in + delta, 0.0, 1.0)
        y_out_bt = out_3[:, 0:1]
        cb_out_up = out_3[:, 1:2]
        cr_out_up = out_3[:, 2:3]

        # Cb/Cr 再下采样回 4:2:0
        cb_out = self.downsample_chroma(cb_out_up)  # (BT,1,H/2,W/2)
        cr_out = self.downsample_chroma(cr_out_up)  # (BT,1,H/2,W/2)

        # 还原回 (B,T,1,H,W) / (B,T,1,H/2,W/2)
        y_out = y_out_bt.view(B, T, 1, H, W)
        cb_out = cb_out.view(B, T, 1, Hc, Wc)
        cr_out = cr_out.view(B, T, 1, Hc, Wc)

        return y_out, cb_out, cr_out


class ValueNetYCbCrPair(nn.Module):
    """
    ValueNet (Critic) – :

      - 上分支: Input Y + Input CbCr↑
      - 下分支: Output Y + Output CbCr↑
      - 两分支分别 Conv3x3 + LeakyReLU + Conv3x3
      - 特征拼接后再 Conv3x3 + AvgPool + MLP -> 标量 Q

    输入张量形状:
        y_in,  cb_in,  cr_in:  (B,T,1,H,W) / (B,T,1,H/2,W/2)
        y_out, cb_out, cr_out: 同上

    输出:
        q: (B,)  视频级 Q 值
    """
    def __init__(self,
                 scale: int = 2,
                 branch_channels: int = 32,
                 fusion_channels: int = 64,
                 num_fusion_blocks: int = 1):
        super().__init__()
        self.scale = scale

        # Cb/Cr 上采样到亮度分辨率
        self.upsample_chroma = nn.Upsample(
            scale_factor=scale, mode="bilinear", align_corners=False
        )

        # 两个对称分支：Conv3x3 + LeakyReLU + Conv3x3
        self.branch_in = nn.Sequential(
            nn.Conv2d(3, branch_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(branch_channels, branch_channels, 3, 1, 1),
        )

        self.branch_out = nn.Sequential(
            nn.Conv2d(3, branch_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(branch_channels, branch_channels, 3, 1, 1),
        )

        # 融合后通道数: 2 * branch_channels
        fusion_in_channels = 2 * branch_channels

        self.fusion_conv = nn.Conv2d(
            fusion_in_channels,
            fusion_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        # 头部: Flatten -> Linear -> ReLU -> Linear
        self.head = nn.Sequential(
            nn.Linear(fusion_channels, fusion_channels),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_channels, 1),
        )

    def _prep(self, y, cb, cr):
        """
        把 Y/Cb/Cr clip 转成 (BT,3,H,W) 方便两分支卷积:
            - Cb/Cr 上采样到亮度分辨率
            - 在通道维拼接: [Y, Cb↑, Cr↑]
        """
        B, T, _, H, W = y.shape
        _, _, _, Hc, Wc = cb.shape

        y_bt = y.view(B * T, 1, H, W)
        cb_bt = cb.view(B * T, 1, Hc, Wc)
        cr_bt = cr.view(B * T, 1, Hc, Wc)

        cb_up = self.upsample_chroma(cb_bt)
        cr_up = self.upsample_chroma(cr_bt)

        x = torch.cat([y_bt, cb_up, cr_up], dim=1)  # (BT,3,H,W)
        return x, B, T

    def forward(self, y_in, cb_in, cr_in, y_out, cb_out, cr_out):
        
        assert y_in.dim() == 5 and y_out.dim() == 5, \
            f"ValueNet expects (B,T,1,H,W), got {y_in.shape}, {y_out.shape}"

        if DEBUG_SHAPES:
            print("[ValueNet] y_in:", y_in.shape, "y_out:", y_out.shape)        
        
        
        # 准备输入 / 输出分支
        x_in, B, T = self._prep(y_in, cb_in, cr_in)
        x_out, _, _ = self._prep(y_out, cb_out, cr_out)

        # 两个分支
        feat_in = self.branch_in(x_in)
        feat_out = self.branch_out(x_out)

        # 融合
        feat_cat = torch.cat([feat_in, feat_out], dim=1)  # (BT,2Cb,H,W)
        feat = self.fusion_conv(feat_cat)  # (BT,Cf,H,W)
        feat = F.leaky_relu(feat, negative_slope=0.1)
        # 全局平均池化 + 时间维平均 -> 视频级特征
        feat = self.pool(feat).view(B, T, -1)  # (B,T,Cf)
        feat = feat.mean(dim=1)                # (B,Cf)

        # MLP 输出标量 Q
        q = self.head(feat).squeeze(1)         # (B,)
        return q
  
    


# ================================================================
# 确定性 Actor-Critic（DDPG 风格，不含 replay buffer）
# ================================================================
class DeterministicACAgentYCbCr:
    """
    确定性 Actor-Critic 智能体 (DDPG风格)

    网络结构:
      - Actor: RAPNetYCbCr - 生成预处理后的YUV帧
      - Critic: ValueNetYCbCrPair - 评估(状态,动作)对的Q值

    训练目标:
      - L_critic = MSE(Q(S,A), R)  # Critic学习预测真实奖励
      - L_actor  = -Q(S, A(S)) + lambda_pixel * L_pixel  # Actor最大化Q值+保持像素相似度

    多GPU策略:
      - Actor在device_actor上 (推荐cuda:0)
      - Critic在device_critic上 (推荐cuda:1,如果可用)
      - critic_for_actor是Critic的副本,放在device_actor上供Actor损失计算使用
    """
    def __init__(self,
                 H_down: int,
                 W_down: int,
                 clip_len: int,
                 lr_actor: float = 1e-4,
                 lr_critic: float = 1e-4,
                 lambda_pixel: float = 0.0005,
                 device_actor: torch.device = DEVICE_ACTOR,
                 device_critic: torch.device = DEVICE_CRITIC):
        self.H_down = H_down
        self.W_down = W_down
        self.clip_len = clip_len
        self.lambda_pixel = lambda_pixel
        self.device_actor = device_actor
        self.device_critic = device_critic

        # -------- Actor 放在 device_actor 上 --------
        self.actor = RAPNetYCbCr().to(self.device_actor)
        
        # -------- 初始化 Actor 权重和偏置为 0 --------
        # 这样网络初始时相当于恒等映射，不会对输入做任何修改
        # 避免随机初始化导致的前期不稳定
        self._init_actor_zero_weights()

        # -------- 训练用 Critic 放在 device_critic 上 --------
        self.critic = ValueNetYCbCrPair().to(self.device_critic)

        # -------- 给 Actor 用的 Critic 副本，放在 device_actor 上 --------
        # 注意: 这个副本的requires_grad需要为True,才能让梯度回传到Actor的输出
        self.critic_for_actor = ValueNetYCbCrPair().to(self.device_actor)
        self._sync_critic_to_actor()

        # ⚠️ 重要修复: critic_for_actor的参数设为requires_grad=False
        # 但网络整体仍需参与前向传播以计算梯度到Actor的输出
        for p in self.critic_for_actor.parameters():
            p.requires_grad = False

        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.mse_loss = nn.MSELoss()
        
    def _sync_critic_to_actor(self):
        """把训练用 Critic 的参数同步到 Actor 侧的 Critic 副本。"""
        self.critic_for_actor.load_state_dict(self.critic.state_dict()) 
        
    def _init_actor_zero_weights(self):
        """
        将 Actor 网络中所有 Conv2d 层的权重初始化为很小的随机值。
        这样网络初始时输出接近 delta=0（近似恒等映射），但梯度可以正常流动。
        
        问题分析：
        - 全0初始化会导致所有中间层激活值为0
        - 反向传播时梯度无法通过0激活值传播
        - 导致Actor梯度一直为0，无法学习
        
        解决方案：
        - 使用很小的随机初始化（std=0.001）
        - 既保持接近恒等映射的初始行为
        - 又能让梯度正常流动
        """
        for module in self.actor.modules():
            if isinstance(module, nn.Conv2d):
                # 使用很小的正态分布初始化权重
                nn.init.normal_(module.weight, mean=0.0, std=0.001)
                # 偏置初始化为0
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)   
    
    def act(self,
            y_state: np.ndarray,
            cb_state: np.ndarray,
            cr_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        根据当前状态 (S) 使用Actor网络输出预处理后的动作 A(S)

        工作流程:
          1. 将numpy状态转换为tensor并移到device_actor
          2. 通过Actor网络(RAPNet)前向传播
          3. 输出预处理后的YUV帧(动作)

        输入:
            y_state:  (T,1,H,W) numpy数组, 归一化到[0,1]
            cb_state: (T,1,H/2,W/2) numpy数组
            cr_state: (T,1,H/2,W/2) numpy数组
        输出:
            y_action, cb_action, cr_action: 同形状的numpy数组
        """
        self.actor.eval()  # 设置为评估模式(禁用Dropout等)

        # 转换为tensor并添加batch维度: (T,1,H,W) -> (1,T,1,H,W)
        y = torch.from_numpy(y_state).float().to(self.device_actor).unsqueeze(0)
        cb = torch.from_numpy(cb_state).float().to(self.device_actor).unsqueeze(0)
        cr = torch.from_numpy(cr_state).float().to(self.device_actor).unsqueeze(0)

        if DEBUG_SHAPES:
            print("[ACT] state shapes:",
                "Y", y_state.shape, "Cb", cb_state.shape, "Cr", cr_state.shape)

        # 无梯度推理
        with torch.no_grad():
            y_a, cb_a, cr_a = self.actor(y, cb, cr)              
            # 再次截断，防止越界
            y_a = torch.clamp(y_a, 0.0, 1.0)
            cb_a = torch.clamp(cb_a, 0.0, 1.0)
            cr_a = torch.clamp(cr_a, 0.0, 1.0)
            # ==========================================================

        # 转回numpy: (1,T,1,H,W) -> (T,1,H,W)
        return (
            y_a.squeeze(0).cpu().numpy(),
            cb_a.squeeze(0).cpu().numpy(),
            cr_a.squeeze(0).cpu().numpy(),
        )

    def update(self,
               y_state: np.ndarray,
               cb_state: np.ndarray,
               cr_state: np.ndarray,
               y_action: np.ndarray,
               cb_action: np.ndarray,
               cr_action: np.ndarray,
               reward: float) -> Tuple[float, float]:
        """
        使用单步经验(S, A, R)更新Actor和Critic网络

        DDPG更新策略:
          1. Critic更新: 最小化 MSE(Q(S,A), R)
             - 使用真实奖励R作为监督信号
             - 在device_critic上计算

          2. Actor更新: 最大化 Q(S, A(S))
             - 通过Critic评估Actor生成的动作质量
             - 添加像素重建损失保持图像保真度
             - L_actor = -Q(S,A(S)) + λ*L_pixel
             - 在device_actor上计算

        Args:
            y/cb/cr_state: 输入状态 (T,1,H,W)
            y/cb/cr_action: Actor输出的动作 (T,1,H,W)
            reward: 环境返回的标量奖励

        Returns:
            (loss_actor, loss_critic): 两个网络的损失值
        """

        self.actor.train()
        self.critic.train()

        # ========= 1) 更新 Critic (在 device_critic 上) =========
        # 将状态和动作移到critic设备
        y_s_c = torch.from_numpy(y_state).float().to(self.device_critic).unsqueeze(0)
        cb_s_c = torch.from_numpy(cb_state).float().to(self.device_critic).unsqueeze(0)
        cr_s_c = torch.from_numpy(cr_state).float().to(self.device_critic).unsqueeze(0)

        y_a_c = torch.from_numpy(y_action).float().to(self.device_critic).unsqueeze(0)
        cb_a_c = torch.from_numpy(cb_action).float().to(self.device_critic).unsqueeze(0)
        cr_a_c = torch.from_numpy(cr_action).float().to(self.device_critic).unsqueeze(0)

        r_c = torch.tensor([reward], dtype=torch.float32, device=self.device_critic)  # (1,)

        # Critic前向: 预测Q(S,A)
        self.optim_critic.zero_grad()
        q_pred = self.critic(y_s_c, cb_s_c, cr_s_c, y_a_c, cb_a_c, cr_a_c)  # (1,)

        # Critic损失: MSE(Q_pred, R_target)
        loss_critic = self.mse_loss(q_pred, r_c)
        loss_critic.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
        self.optim_critic.step()

        # Critic更新完后，同步参数到Actor侧的副本
        self._sync_critic_to_actor()

        # ========= 2) 更新 Actor (在 device_actor 上) =========
        self.actor.train()
        self.critic_for_actor.eval()   # Critic副本仅用于前向,不更新参数

        # 将状态移到actor设备
        y_s_a = torch.from_numpy(y_state).float().to(self.device_actor).unsqueeze(0)
        cb_s_a = torch.from_numpy(cb_state).float().to(self.device_actor).unsqueeze(0)
        cr_s_a = torch.from_numpy(cr_state).float().to(self.device_actor).unsqueeze(0)

        self.optim_actor.zero_grad()

        # Actor前向: 生成当前策略下的动作 A'(S)
        y_a2, cb_a2, cr_a2 = self.actor(y_s_a, cb_s_a, cr_s_a)

        # 使用Critic副本评估Actor的输出: Q(S, A'(S))
        # 梯度会回传到y_a2/cb_a2/cr_a2,进而更新Actor参数
        q_for_actor = self.critic_for_actor(y_s_a, cb_s_a, cr_s_a, y_a2, cb_a2, cr_a2)
        
        # 【新增】核心诊断代码
        # 计算 Q 对 y_a2 (亮度通道动作) 的梯度
        # retain_graph=True 是必须的，因为后面还要做 loss_actor.backward()
        grads = torch.autograd.grad(
            outputs=q_for_actor.sum(), 
            inputs=y_a2, 
            retain_graph=True, 
            create_graph=False, 
            allow_unused=True
        )[0]
        
        # # 记录梯度范数
        if grads is not None:
            grad_norm = grads.norm().item()
            # 这里的 writer 需要从外部传入，或者你可以在 agent 里存一个 writer 引用
            # 为了方便，这里我们存到 self 变量里，在外面 log
            self.last_dq_da_norm = grad_norm
        else:
            self.last_dq_da_norm = 0.0
        
            
        # --------- helpers (local) ----------
        def blur5x5_torch(x4):
            k = torch.ones((1,1,5,5), device=x4.device, dtype=x4.dtype) / 25.0
            x4 = F.pad(x4, (2,2,2,2), mode="reflect")
            return F.conv2d(x4, k)

        def sobel_edge_mask(x4, k=10.0, floor=0.05):
            # x4: (N,1,H,W)
            edge = sobel_edge_mag(x4)                 # already defined in your file
            # normalize edge roughly (avoid exploding / tiny)
            edge_n = edge / (edge.mean() + 1e-6)
            # squash to [0,1] like a soft mask
            E = 1.0 - torch.exp(-k * edge_n)
            return E.clamp(min=floor, max=1.0)

        # flatten (B,T,1,H,W)->(BT,1,H,W) for conv
        B, T, _, H, W = y_s_a.shape
        y_s_bt = y_s_a.view(B*T, 1, H, W)
        y_a_bt = y_a2.view(B*T, 1, H, W)

        delta = y_a_bt - y_s_bt

        # ---- Low-frequency structure term: LP(y_a) close to LP(y_s) ----
        lp_s = blur5x5_torch(y_s_bt)
        lp_a = blur5x5_torch(y_a_bt)
        loss_low = F.l1_loss(lp_a, lp_s)

        # ---- High-frequency suppression (only non-edge regions) ----
        hp_delta = delta - blur5x5_torch(delta)       # high-pass of the modification
        E = sobel_edge_mask(y_s_bt, k=10.0, floor=0.05)
        non_edge = (1.0 - E)

        # penalize HF changes away from edges
        loss_hf = torch.mean(torch.abs(hp_delta) * non_edge)

        # ---- Edge focusing: encourage modifications to happen on edges ----
        # (maximize |delta| on edges relative to total, implemented as negative reward / loss)
        edge_focus = torch.sum(torch.abs(delta) * E) / (torch.sum(torch.abs(delta)) + 1e-8)
        loss_edge_focus = 1.0 - edge_focus
        

        # ---- keep mild TV only in non-edge (optional, but keep small) ----
        tv_loss = total_variation_loss_masked(y_a2, weight=1.0, edge_k=30.0, edge_floor=0.05)

        # ---- (optional) tiny identity to prevent drift ----
        loss_id = F.l1_loss(y_a2, y_s_a)

        # ---- final actor loss ----
        # Q should dominate; others are shaping
        w_low = 0.5
        w_hf  = 10.0
        w_edge= 0.2
        w_tv  = 10
        w_id  = 0.01

        diff = y_a2 - y_s_a
        loss_mse = (diff ** 2).mean()  # 或者 F.mse_loss(y_a2, y_s_a)
        w_mse = 5
        # loss_actor = -q_for_actor.mean() \
        #             + w_low  * loss_low \
        #             + w_hf   * loss_hf \
        #             + w_edge * loss_edge_focus \
        #             + w_tv   * tv_loss \
        #             + w_id   * loss_id
        loss_actor = -q_for_actor.mean() + w_mse * loss_mse + w_hf * loss_hf



        # --- 【核心】动力学诊断 (在你 Backward 之前计算) ---
        # 计算 Critic 想要 Actor 往哪走 (推力)
        # 这里的 grads 是 Q 对 Actor输出 (y_a2) 的梯度
        grads_q = torch.autograd.grad(q_for_actor.mean(), y_a2, retain_graph=True, create_graph=False)[0]
        norm_force_q = grads_q.norm().item()
        
        # 计算 MSE + HF 想要 Actor 往哪走 (阻力/Drag)
        # 阻力 = 所有正则项 (MSE + HF) 的梯度
        # 注意：我们要算的是 total_penalty = w_mse * loss_mse + w_hf * loss_hf 的梯度
        loss_penalty = w_mse * loss_mse + w_hf * loss_hf
        grads_drag = torch.autograd.grad(loss_penalty, y_a2, retain_graph=True, create_graph=False)[0]
        norm_force_drag = grads_drag.norm().item()
        
        # 存入 self 供外部读取
        self.last_force_q = norm_force_q
        self.last_force_mse = norm_force_drag # 借用变量名，实际是 Total Drag
        self.last_force_ratio = norm_force_q / (norm_force_drag + 1e-9) # 推阻比
        loss_actor.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.optim_actor.step()


        if DEBUG_SHAPES:
            print(
                "[UPDATE] reward:", float(reward),
                "Q_pred:", float(q_pred.detach().cpu().item()),
            )
        
        self.last_loss_actor = loss_actor.detach()
        self.last_loss_critic = loss_critic.detach()
        # critic 更新后
        self.last_q_pred = float(q_pred.detach().cpu().item())
        self.last_td_error = float(torch.abs(q_pred.detach() - r_c).cpu().item())

        # actor 更新后（q_for_actor 是 tensor (B,)）
        self.last_q_actor = float(q_for_actor.mean().detach().cpu().item())
        
        # (log scalars if you want)
        self.last_loss_actor = loss_actor.detach()
        self.last_low_loss = float(loss_low.detach().cpu().item())
        self.last_hf_loss = float(loss_hf.detach().cpu().item())
        self.last_edge_focus = float(edge_focus.detach().cpu().item())
        self.last_tv_loss = float(tv_loss.detach().cpu().item())
        self.last_id_loss = float(loss_id.detach().cpu().item())
        self.last_mse_loss = float(loss_mse.detach().cpu().item()) * w_mse
        return float(loss_actor.item()), float(loss_critic.item())


# ================================================================
# 基于 hevc_nvenc constqp 的环境
# ================================================================
class NvencClipEnvQP:
    """
    视频编码强化学习环境 - 基于HEVC_NVENC

    任务目标:
      学习一个预处理网络(RAPNet),在编码前优化视频帧,
      使得编码后的Rate-Distortion性能优于baseline

    环境工作流程:
      1. reset(): 从视频库随机采样一个clip (16帧)
         - 读取1080p YUV420原始帧
         - 下采样到编码分辨率(540p)作为状态S

      2. step(action): 接收预处理后的帧(动作A)
         - 对每个QP(22,27,32,37,42)进行HEVC编码
         - 解码并上采样回1080p
         - 计算RD奖励: R = α*ΔV + β*ΔR
           * ΔV: 质量提升(LPIPS)
           * ΔR: 码率节省

    状态空间(State):
      - 下采样后的YUV420帧: Y(T,1,H/2,W/2), Cb/Cr(T,1,H/4,W/4)

    动作空间(Action):
      - 预处理后的YUV420帧(同状态维度)
      - 由Actor网络(RAPNet)生成

    奖励(Reward):
      - RD增益: R = α*(V_cur-V_ref)/|V_ref| + β*(R_ref-R_cur)/R_ref
      - 对5个QP点求平均
    """

    def __init__(self,
                 video_meta_list: List[Dict],
                 rd_curve_json: str,
                 work_dir: str = "./rl_nvenc_tmp",
                 qp_list: List[int] = None,
                 clip_len: int = 16,
                 fps: int = 120,
                 width: int = 1920,
                 height: int = 1080,
                 downscale: int = 2,
                 ffmpeg_bin: str = "ffmpeg",
                 alpha_q: float = 5,
                 beta_r: float = 5):
        self.video_meta_list = video_meta_list
        self.rd_table = json.load(open(rd_curve_json, "r", encoding="utf-8")) \
            if os.path.exists(rd_curve_json) else {}

        self.work_dir = work_dir
        os.makedirs(self.work_dir, exist_ok=True)

        self.qp_list = qp_list or [37,40,42,45,47]
        self.clip_len = clip_len
        self.fps = fps
        self.W = width
        self.H = height
        self.downscale = downscale
        self.ffmpeg_bin = ffmpeg_bin
        self.alpha_q = alpha_q
        self.beta_r = beta_r

        self.H_down = self.H // self.downscale
        self.W_down = self.W // self.downscale

        self.current_video_id = None
        self.ref_clip_1080p = None
        self.state_y = None
        self.state_cb = None
        self.state_cr = None

    def reset(self):
        """
        重置环境,开始新的episode

        工作流程:
          1. 从视频库随机选择一个视频
          2. 从该视频随机采样一个clip(16帧)
          3. 读取1080p YUV420原始帧
          4. 下采样到编码分辨率(540p)作为状态
          5. 保存1080p BGR帧用于后续LPIPS计算

        Returns:
            y_ds, cb_ds, cr_ds: 下采样后的YUV420状态
              - y_ds:  (T,1,H/2,W/2)
              - cb_ds: (T,1,H/4,W/4)
              - cr_ds: (T,1,H/4,W/4)
            info: {'video_id': str}
        """
        meta = random.choice(self.video_meta_list)
        self.current_video_id = meta["id"]

        # 1) 直接从 .yuv 文件读取 1080p 的 Y/Cb/Cr 平面
        y_1080, cb_1080, cr_1080 = load_clip_yuv_planes(meta, self.clip_len)

        # 2) 保存参考 1080p BGR（只用于 LPIPS 计算）
        self.ref_clip_1080p = yuv420_planes_to_bgr_clip(y_1080, cb_1080, cr_1080)

        # 3) 下采样到编码分辨率 (例如 540p)
        y_ds, cb_ds, cr_ds = downsample_yuv420_clip(
            y_1080, cb_1080, cr_1080, scale=self.downscale
        )

        # 保存状态供step()使用
        self.state_y = y_ds
        self.state_cb = cb_ds
        self.state_cr = cr_ds

        # 确保数值范围在[0,1]
        y_ds = np.clip(y_ds, 0.0, 1.0)
        cb_ds = np.clip(cb_ds, 0.0, 1.0)
        cr_ds = np.clip(cr_ds, 0.0, 1.0)

        info = {"video_id": self.current_video_id}
        if DEBUG_SHAPES:
            print("[RESET] video_id:", self.current_video_id)
            print("  y_1080:", y_1080.shape, "cb_1080:", cb_1080.shape, "cr_1080:", cr_1080.shape)
            print("  y_ds:", y_ds.shape, "cb_ds:", cb_ds.shape, "cr_ds:", cr_ds.shape)
            print("  y_ds range:", float(y_ds.min()), float(y_ds.max()))
        return y_ds, cb_ds, cr_ds, info

    def step(self,
             y_action: np.ndarray,
             cb_action: np.ndarray,
             cr_action: np.ndarray):
        """
        执行动作(预处理后的帧),计算RD奖励

        工作流程:
          1. 将预处理后的YUV420帧写入临时文件
          2. 对每个QP点(22,27,32,37,42)进行HEVC编码
             a) 使用HEVC_NVENC编码
             b) 解码并上采样回1080p
             c) 计算质量(LPIPS)和码率
             d) 与baseline对比,计算RD增益
          3. 对所有QP点的RD增益求平均作为奖励

        Args:
            y_action:  (T,1,H_down,W_down) 预处理后的亮度分量
            cb_action: (T,1,H_down/2,W_down/2) 预处理后的Cb分量
            cr_action: (T,1,H_down/2,W_down/2) 预处理后的Cr分量

        Returns:
            next_state: (y, cb, cr) - 与输入状态相同(单步episode)
            reward: float - 平均RD增益
            done: bool - 总是True(单步episode)
            info: dict - 包含详细的RD统计信息
              {
                'R_cur': [r1,r2,r3,r4,r5],  # 当前码率(kbps)
                'V_cur': [v1,v2,v3,v4,v5],  # 当前质量(-LPIPS)
                'R_ref': [r1,r2,r3,r4,r5],  # baseline码率
                'V_ref': [v1,v2,v3,v4,v5],  # baseline质量
                'qp_list': [27, 32, 37, 40, 42]
              }
        """
        T = self.clip_len
        H_down = self.H_down
        W_down = self.W_down
        Hc = H_down // 2
        Wc = W_down // 2

        # 验证动作维度
        assert y_action.shape == (T, 1, H_down, W_down), \
            f"Expected y_action shape ({T},1,{H_down},{W_down}), got {y_action.shape}"
        assert cb_action.shape == (T, 1, Hc, Wc), \
            f"Expected cb_action shape ({T},1,{Hc},{Wc}), got {cb_action.shape}"
        assert cr_action.shape == (T, 1, Hc, Wc), \
            f"Expected cr_action shape ({T},1,{Hc},{Wc}), got {cr_action.shape}"

        if DEBUG_SHAPES:
            print("[STEP] action shapes:",
                "Y", y_action.shape, "Cb", cb_action.shape, "Cr", cr_action.shape)
            print("       action range Y:", float(y_action.min()), float(y_action.max()))

        rewards_per_qp = []
        delta_v_list, delta_r_list = [], []
        R_cur_list, V_cur_list = [], []
        R_ref_list, V_ref_list = [], []

        # 使用临时目录管理中间文件
        with tempfile.TemporaryDirectory(dir=self.work_dir) as tmpdir:
            yuv_in = os.path.join(tmpdir, "preproc.yuv")

            # 将预处理结果写成YUV文件,供所有QP复用
            yuv_bytes = ycbcr420_clip_to_yuv420_bytes(y_action, cb_action, cr_action)
            with open(yuv_in, "wb") as f:
                f.write(yuv_bytes)

            # 对每个QP点进行编码-解码-评估
            for qp in self.qp_list:
                bitstream = os.path.join(tmpdir, f"bitstream_qp{qp}.hevc")
                yuv_dec = os.path.join(tmpdir, f"decoded_qp{qp}.yuv")
                log_path = os.path.join(tmpdir, f"enc_qp{qp}.log")

                # === 编码阶段 ===
                encode_cmd = [
                    self.ffmpeg_bin,
                    "-y",
                    "-f", "rawvideo",
                    "-pix_fmt", "yuv420p",
                    "-s:v", f"{W_down}x{H_down}",
                    "-r", str(self.fps),
                    "-i", yuv_in,

                    "-c:v", "hevc_nvenc",
                    "-rc", "constqp",      # 恒定QP模式
                    "-qp", str(qp),        # QP值
                    "-preset", "p4",       # 编码预设(p4=medium)
                    "-profile:v", "main",
                    # "-bf", "3",            # B帧数量
                    # "-rc-lookahead", "32", # 前瞻帧数
                    # "-spatial-aq", "1",    # 空间自适应量化
                    # "-aq-strength", "8",   # AQ强度
                    "-pix_fmt", "yuv420p",

                    bitstream,
                ]

                # 捕获stderr用于解析码率
                with open(log_path, "w") as f_log:
                    subprocess.run(encode_cmd, check=True,
                                   stdout=subprocess.DEVNULL, stderr=f_log)

                # === 解码阶段 ===
                decode_cmd = [
                    self.ffmpeg_bin,
                    "-y",
                    "-i", bitstream,
                    "-f", "rawvideo",
                    "-pix_fmt", "yuv420p",
                    yuv_dec,
                ]
                subprocess.run(decode_cmd, check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                # === 质量评估 ===
                # 读取解码后的540p YUV
                y_dec, cb_dec, cr_dec = read_yuv420_clip_to_planes(
                    yuv_dec, T, H_down, W_down
                )
                # 上采样回1080p用于LPIPS计算
                y_dec_up, cb_dec_up, cr_dec_up = upsample_yuv420_clip(
                    y_dec, cb_dec, cr_dec, scale=self.downscale
                )
                # 转BGR格式(LPIPS需要)
                dec_clip_1080p = yuv420_planes_to_bgr_clip(
                    y_dec_up, cb_dec_up, cr_dec_up
                )

                # === 码率计算 ===
                # 从FFmpeg日志解析纯视频流大小(排除容器开销)
                video_size_kb = parse_ffmpeg_log_for_video_size(log_path)
                duration_sec = T / float(self.fps)

                if video_size_kb > 0:
                    # video_size_kb (kB) * 1024 * 8 -> bits
                    total_bits = video_size_kb * 1024 * 8
                    bitrate_kbps = total_bits / 1000.0 / duration_sec
                else:
                    # Fallback: 使用文件大小估算(包含容器开销)
                    bitrate_kbps = os.path.getsize(bitstream) * 8.0 / duration_sec / 1000.0

                # 计算LPIPS质量分数(取负值,越大越好)
                V_cur = compute_quality_metric(self.ref_clip_1080p, dec_clip_1080p)

                if DEBUG_SHAPES:
                  print(f"    [QP {qp}] bitrate={bitrate_kbps:.1f} kbps, V_cur={V_cur:.4f}")

                # === 与baseline对比 ===
                ref_entry = self.rd_table[self.current_video_id][str(qp)]
                R_ref = ref_entry["R"]  # baseline码率
                V_ref = ref_entry["V"]  # baseline质量

                # # RD增益计算
                # # ΔV = (V_cur - V_ref) / |V_ref|  # 质量提升比例
                # # ΔR = (R_ref - R_cur) / R_ref     # 码率节省比例
                # delta_v = (V_cur - V_ref) / (abs(V_ref) + 1e-6)
                # delta_r = (R_ref - bitrate_kbps) / (R_ref + 1e-6)

                # # 综合RD奖励: α*ΔV + β*ΔR
                # reward_qp = self.alpha_q * delta_v + self.beta_r * delta_r
                
                # ================= v18 修改后代码 (非对称奖励) =================
                delta_v = (V_cur - V_ref) / (abs(V_ref) + 1e-6)
                delta_r = (R_ref - bitrate_kbps) / (R_ref + 1e-6)
                

                reward_final = 100 * delta_r + 200 * delta_v
                rewards_per_qp.append(reward_final)          
                delta_v_list.append(delta_v)
                delta_r_list.append(delta_r)
                R_cur_list.append(bitrate_kbps)
                V_cur_list.append(V_cur)
                R_ref_list.append(R_ref)
                V_ref_list.append(V_ref)

        # 对所有QP点的奖励求平均
        reward = float(np.mean(rewards_per_qp))
        info = {
            "qp_list": self.qp_list,
            "reward_qp": rewards_per_qp,
            "delta_v": delta_v_list,
            "delta_r": delta_r_list,
            "reward_mean": reward,
            "R_cur": R_cur_list,
            "V_cur": V_cur_list,
            "R_ref": R_ref_list,
            "V_ref": V_ref_list,
        }

        done = True  # 单步episode
        next_state = (self.state_y, self.state_cb, self.state_cr)
        return next_state, reward, done, info



# ================================================================
# 修正后的 Baseline 构建函数
# ================================================================
def build_baseline_rd_json_qp_fullvideo(
        video_meta_list: List[Dict],
        output_json_path: str,
        qp_list: List[int] = None,
        fps: int = 120,          # 与源视频一致，Beauty 是 120fps
        downscale: int = 2,
        ffmpeg_bin: str = "ffmpeg",
        work_dir: str = "ElegantRL-master/my_RL/rd_baseline_full_tmp"):
    """
    基于“整条视频”的 Baseline RD 构建函数：

    对于每个视频 vid、每个 QP：
      1) 读取整个 1080p YUV420 视频 (T_all 帧)
      2) 在 YUV420 平面上整体下采样到编码分辨率 (H/downscale, W/downscale)
      3) 写入一个 540p YUV (T_all 帧)，用 hevc_nvenc constqp 编码整条视频
      4) 解析 FFmpeg 日志，得到整条视频的平均码率 R_ref(qp)
      5) 解码成 540p YUV，再上采样回 1080p，转 BGR
      6) 对整条视频 (T_all 帧) 计算 LPIPS 平均值 (我们已有逐帧实现)，取负号作为 V_ref(qp)
    最终保存 JSON: rd_dict[vid][qp] = {"R": R_ref, "V": V_ref}
    """
    qp_list =  [37,40,42,45,47] if qp_list is None else qp_list

    # 清理 / 创建工作目录
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    rd_dict: Dict[str, Dict[str, Dict[str, float]]] = {}

    print("========== [Baseline Build - Full Video] ==========")
    print(f"QP list: {qp_list}")
    print(f"work_dir: {work_dir}")
    print(f"LPIPS device: {device}")

    for meta in video_meta_list:
        vid = meta["id"]
        path = meta["path"]
        W = int(meta["width"])
        H = int(meta["height"])
        fps_meta = int(meta.get("fps", fps))

        print(f"\n>>> Processing Video (full): {vid}")
        print(f"    path={path}, W={W}, H={H}")

        # 1) 计算总帧数
        frame_bytes = int(W * H * 3 / 2)
        file_size = os.path.getsize(path)
        total_frames = file_size // frame_bytes
        print(f"    total_frames={total_frames}")

        # 2) 读取整条 1080p YUV420 -> (T_all,1,H,W)/(T_all,1,H/2,W/2)
        y_1080, cb_1080, cr_1080 = read_yuv420_clip_to_planes(
            path, total_frames, H, W
        )

        # 3) 构造整条视频的 1080p BGR 参考序列 (用于 LPIPS)
        ref_clip_bgr = yuv420_planes_to_bgr_clip(y_1080, cb_1080, cr_1080)
        print(f"    ref_clip_bgr shape={ref_clip_bgr.shape}")

        # 4) 在 YUV420 平面上整体下采样到编码分辨率
        H_down = H // downscale
        W_down = W // downscale
        y_ds, cb_ds, cr_ds = downsample_yuv420_clip(
            y_1080, cb_1080, cr_1080, scale=downscale
        )
        # Y_ds: (T_all,1,H_down,W_down), Cb/Cr: (T_all,1,H_down/2,W_down/2)
        yuv_bytes_input = ycbcr420_clip_to_yuv420_bytes(y_ds, cb_ds, cr_ds)

        # 5) 为该视频准备结果字典
        rd_dict[vid] = {}

        # 把整条 540p 输入写入一个临时 yuv 文件 (供所有 QP 复用)
        input_yuv_path = os.path.join(work_dir, f"{vid}_full_in.yuv")
        with open(input_yuv_path, "wb") as f:
            f.write(yuv_bytes_input)

        # 6) 对每个 QP 做一次整视频编码 / 解码 / LPIPS
        for qp in qp_list:
            print(f"    [QP {qp}] encoding full video ...")

            bitstream_path = os.path.join(work_dir, f"{vid}_full_q{qp}.hevc")
            dec_yuv_path = os.path.join(work_dir, f"{vid}_full_q{qp}_dec.yuv")
            log_path = os.path.join(work_dir, f"{vid}_full_q{qp}.log")

            # --- 6.1 编码整条 540p YUV ---
            encode_cmd = [
                ffmpeg_bin, "-y",
                "-f", "rawvideo", "-pix_fmt", "yuv420p",
                "-s:v", f"{W_down}x{H_down}",
                "-r", str(fps_meta),
                "-i", input_yuv_path,
                "-c:v", "hevc_nvenc",
                "-rc", "constqp",
                "-qp", str(qp),
                "-preset", "p4",
                "-profile:v", "main",
                # "-bf", "3",
                # "-rc-lookahead", "32",
                # "-spatial-aq", "1",
                # "-aq-strength", "8",
                "-pix_fmt", "yuv420p",
                bitstream_path,
            ]
            with open(log_path, "w") as f_log:
                subprocess.run(encode_cmd, check=True,
                               stdout=subprocess.DEVNULL, stderr=f_log)

            # --- 6.2 解析整条视频码率 R_ref(qp) ---
            video_size_kb = parse_ffmpeg_log_for_video_size(log_path)
            duration_sec = total_frames / float(fps_meta)
            if video_size_kb > 0:
                total_bits = video_size_kb * 1024 * 8
                bitrate_kbps = total_bits / 1000.0 / duration_sec
            else:
                # fallback: 按 bitstream 文件大小估算
                bitrate_kbps = os.path.getsize(bitstream_path) * 8.0 / 1000.0 / duration_sec

            # --- 6.3 解码整条视频到 540p YUV420 ---
            decode_cmd = [
                ffmpeg_bin, "-y",
                "-i", bitstream_path,
                "-f", "rawvideo",
                "-pix_fmt", "yuv420p",
                dec_yuv_path,
            ]
            subprocess.run(decode_cmd, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # --- 6.4 从解码 YUV 中读取全帧，升到 1080p，再转 BGR ---
            y_dec, cb_dec, cr_dec = read_yuv420_clip_to_planes(
                dec_yuv_path, total_frames, H_down, W_down
            )
            y_up, cb_up, cr_up = upsample_yuv420_clip(
                y_dec, cb_dec, cr_dec, scale=downscale
            )
            dec_clip_bgr = yuv420_planes_to_bgr_clip(y_up, cb_up, cr_up)

            # --- 6.5 对整条视频计算 LPIPS 平均值 ---
            V_ref = compute_quality_metric(ref_clip_bgr, dec_clip_bgr)

            rd_dict[vid][str(qp)] = {
                "R": float(bitrate_kbps),
                "V": float(V_ref),
            }
            print(f"        bitrate={bitrate_kbps:.2f} kbps, V_ref={V_ref:.4f}")

            # 清理 QP 的中间文件（可选）
            for p in [bitstream_path, dec_yuv_path, log_path]:
                if os.path.exists(p):
                    os.remove(p)

        # 可以选择保留 input_yuv_path，或者删除节省空间
        if os.path.exists(input_yuv_path):
            os.remove(input_yuv_path)

    # 7) 保存最终 JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(rd_dict, f, indent=2, ensure_ascii=False)

    print(f"\n[Baseline-Full] Saved to {output_json_path}")


def build_baseline_rd_json_qp_clips(
        video_meta_list: List[Dict],
        output_json_path: str,
        samples_per_video: int = 30,     # 新增参数：每个视频采样多少个 Clip
        clip_len: int = 16,              # 必须与训练时的 clip_len 严格一致
        qp_list: List[int] = None,
        fps: int = 120,
        downscale: int = 2,
        ffmpeg_bin: str = "ffmpeg",
        work_dir: str = "ElegantRL-master/my_RL/rd_baseline_clips_tmp"):
    """
    基于“随机 Clip 采样”的 Baseline RD 构建函数（修正版）：

    【为什么改用 Clip 采样？】
    训练时我们只切取 16 帧进行编码，这会导致第一个帧必须是 I 帧。
    对于短 Clip，I 帧的巨大比特数会被平摊到很少的帧上，导致平均码率远高于长视频。
    如果用“整条视频”做 Baseline，训练时的码率永远会比 Baseline 高，导致 Reward 恒为负。
    本函数模拟训练过程，随机采样 Clip 计算 RD，从而建立“公平”的对照组。

    流程：
    对于每个视频 vid：
      1) 循环采样 samples_per_video 次：
         a. 随机读取 16 帧 (1080p)
         b. 下采样到 540p
         c. 编码这个短 Clip (16帧)
         d. 计算该 Clip 的实际码率 (包含 I 帧开销) 和 LPIPS
      2) 对所有采样点的 R 和 V 取平均值，作为该 QP 下的 Baseline
    最终保存 JSON: rd_dict[vid][qp] = {"R": R_mean, "V": V_mean}
    """
    qp_list = qp_list or [37, 40, 42, 45, 47]
    qp_list =[37, 40, 42, 45, 47] # 仅测试 QP=45
    # 清理 / 创建工作目录
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    rd_dict: Dict[str, Dict[str, Dict[str, float]]] = {}

    print("========== [Baseline Build - Random Clips Sampling] ==========")
    print(f"QP list: {qp_list}")
    print(f"Samples per video: {samples_per_video}")
    print(f"Clip length: {clip_len}")
    print(f"work_dir: {work_dir}")

    for meta in video_meta_list:
        vid = meta["id"]
        W = int(meta["width"])
        H = int(meta["height"])
        H_down = H // downscale
        W_down = W // downscale

        print(f"\n>>> Processing Video: {vid} (sampling {samples_per_video} clips)")

        # 初始化统计容器
        # stats[str(qp)]["R"] 存放该 QP 下所有采样 clip 的码率
        stats = {str(qp): {"R": [], "V": []} for qp in qp_list}

        for i in range(samples_per_video):
            # 1) 随机读取一个 Clip (16帧, 1080p YUV)
            # 注意：这里调用 load_clip_yuv_planes (它内部处理随机 seek)
            y_1080, cb_1080, cr_1080 = load_clip_yuv_planes(meta, clip_len)

            # 2) 构造 1080p BGR 参考 (用于 LPIPS)
            ref_clip_bgr = yuv420_planes_to_bgr_clip(y_1080, cb_1080, cr_1080)

            # 3) 下采样到编码分辨率 (540p)
            y_ds, cb_ds, cr_ds = downsample_yuv420_clip(
                y_1080, cb_1080, cr_1080, scale=downscale
            )
            
            # 准备输入文件 (Clip 级别)
            yuv_bytes_input = ycbcr420_clip_to_yuv420_bytes(y_ds, cb_ds, cr_ds)
            input_yuv_path = os.path.join(work_dir, f"{vid}_sample_{i}_in.yuv")
            with open(input_yuv_path, "wb") as f:
                f.write(yuv_bytes_input)

            # 4) 遍历所有 QP 进行编码测试
            for qp in qp_list:
                bitstream_path = os.path.join(work_dir, f"{vid}_s{i}_q{qp}.hevc")
                dec_yuv_path = os.path.join(work_dir, f"{vid}_s{i}_q{qp}_dec.yuv")
                log_path = os.path.join(work_dir, f"{vid}_s{i}_q{qp}.log")

                # --- 编码 (模拟 Env 中的 NVENC 设置) ---
                encode_cmd = [
                    ffmpeg_bin, "-y",
                    "-f", "rawvideo", "-pix_fmt", "yuv420p",
                    "-s:v", f"{W_down}x{H_down}",
                    "-r", str(fps),
                    "-i", input_yuv_path,
                    "-c:v", "hevc_nvenc",
                    "-rc", "constqp",
                    "-qp", str(qp),
                    "-preset", "p4",
                    "-profile:v", "main",
                    # "-bf", "3",
                    # "-rc-lookahead", "32",
                    # "-spatial-aq", "1",
                    # "-aq-strength", "8",
                    "-pix_fmt", "yuv420p",
                    bitstream_path,
                ]
                with open(log_path, "w") as f_log:
                    subprocess.run(encode_cmd, check=True,
                                   stdout=subprocess.DEVNULL, stderr=f_log)

                # --- 解析码率 (关键：时间基准是 clip_len) ---
                video_size_kb = parse_ffmpeg_log_for_video_size(log_path)
                duration_sec = clip_len / float(fps)  # 必须是 16帧的时间，而非整视频

                if video_size_kb > 0:
                    total_bits = video_size_kb * 1024 * 8
                    bitrate_kbps = total_bits / 1000.0 / duration_sec
                else:
                    bitrate_kbps = os.path.getsize(bitstream_path) * 8.0 / 1000.0 / duration_sec

                # --- 解码 & 上采样 & 计算 LPIPS ---
                decode_cmd = [
                    ffmpeg_bin, "-y",
                    "-i", bitstream_path,
                    "-f", "rawvideo",
                    "-pix_fmt", "yuv420p",
                    dec_yuv_path,
                ]
                subprocess.run(decode_cmd, check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                y_dec, cb_dec, cr_dec = read_yuv420_clip_to_planes(
                    dec_yuv_path, clip_len, H_down, W_down
                )
                y_up, cb_up, cr_up = upsample_yuv420_clip(
                    y_dec, cb_dec, cr_dec, scale=downscale
                )
                dec_clip_bgr = yuv420_planes_to_bgr_clip(y_up, cb_up, cr_up)

                V_val = compute_quality_metric(ref_clip_bgr, dec_clip_bgr)

                # 记录该样本数据
                stats[str(qp)]["R"].append(bitrate_kbps)
                stats[str(qp)]["V"].append(V_val)

                # 清理中间文件
                for p in [bitstream_path, dec_yuv_path, log_path]:
                    if os.path.exists(p): os.remove(p)

            # 打印进度 (每5个sample打印一次)
            if (i + 1) % 5 == 0:
                print(f"    Sample {i+1}/{samples_per_video} done.")
            
            if os.path.exists(input_yuv_path): os.remove(input_yuv_path)

        # 5) 汇总该视频的所有样本取平均
        rd_dict[vid] = {}
        for qp in qp_list:
            avg_r = float(np.mean(stats[str(qp)]["R"]))
            avg_v = float(np.mean(stats[str(qp)]["V"]))
            rd_dict[vid][str(qp)] = {"R": avg_r, "V": avg_v}
            print(f"  -> QP {qp}: Avg R={avg_r:.2f}, Avg V={avg_v:.4f}")

    # 6) 保存最终 JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(rd_dict, f, indent=2, ensure_ascii=False)

    print(f"\n[Baseline-Clips] Saved to {output_json_path}")
    
def build_baseline_rd_json_qp_clips_ffmpeg(
        video_meta_list: List[Dict],
        output_json_path: str,
        samples_per_video: int = 30,     
        clip_len: int = 16,              
        qp_list: List[int] = None,
        fps: int = 120,
        downscale: int = 2,
        ffmpeg_bin: str = "ffmpeg",
        work_dir: str = "ElegantRL-master/my_RL/rd_baseline_clips_tmp"):
    """
    修改版：完全对齐 Stage 3 评测标准，并增加调试打印。
    """
    qp_list = qp_list or [37,40,42,45,47]
    
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    # 1. 核心修改：强制初始化为 AlexNet (与 Stage 3 对齐)
    global _lpips_model
    print(f"[*] Loading LPIPS AlexNet for Stage 3 alignment...")
    _lpips_model = lpips.LPIPS(net='alex').to(LPIPS_DEVICE)
    _lpips_model.eval()

    rd_dict = {}

    for meta in video_meta_list:
        vid = meta["id"]
        W, H = int(meta["width"]), int(meta["height"])
        H_down, W_down = H // downscale, W // downscale
        
        print(f"\n{'='*60}")
        print(f"VIDEO: {vid} | Target: 1080p -> {H_down}p")
        print(f"{'='*60}")
        
        stats = {str(qp): {"R": [], "V": []} for qp in qp_list}

        for i in range(samples_per_video):
            # 随机采样 16 帧
            y_1080, cb_1080, cr_1080 = load_clip_yuv_planes(meta, clip_len)
            ref_clip_bgr = yuv420_planes_to_bgr_clip(y_1080, cb_1080, cr_1080)

            # 下采样 (这里作为编码输入，保持 INTER_LANCZOS4)
            y_ds, cb_ds, cr_ds = downsample_yuv420_clip(y_1080, cb_1080, cr_1080, scale=downscale)
            yuv_bytes_input = ycbcr420_clip_to_yuv420_bytes(y_ds, cb_ds, cr_ds)
            input_yuv_path = os.path.join(work_dir, f"{vid}_s{i}_in.yuv")
            with open(input_yuv_path, "wb") as f:
                f.write(yuv_bytes_input)

            print(f"  [Sample {i+1:02d}/{samples_per_video}]", end=" ", flush=True)

            for qp in qp_list:
                bitstream_path = os.path.join(work_dir, f"s{i}_q{qp}.hevc")
                log_path = os.path.join(work_dir, f"s{i}_q{qp}.log")
                yuv_dec_up = os.path.join(work_dir, f"s{i}_q{qp}_up1080.yuv")

                # --- 编码 ---
                encode_cmd = [
                    ffmpeg_bin, "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p",
                    "-s:v", f"{W_down}x{H_down}", "-r", str(fps), "-i", input_yuv_path,
                    "-c:v", "hevc_nvenc", "-rc", "constqp", "-qp", str(qp),
                    "-preset", "p4", "-profile:v", "main", # "-bf", "3",
                    # "-rc-lookahead", "32", # "-spatial-aq", "1", # "-aq-strength", "8",
                    "-pix_fmt", "yuv420p", bitstream_path
                ]
                with open(log_path, "w") as f_log:
                    subprocess.run(encode_cmd, check=True, stdout=subprocess.DEVNULL, stderr=f_log)

                # --- 核心对齐：FFmpeg 上采样 (Stage 3 标准) ---
                decode_up_cmd = [
                    ffmpeg_bin, "-y", "-i", bitstream_path,
                    "-vf", "scale=1920:1080:flags=lanczos",
                    "-f", "rawvideo", "-pix_fmt", "yuv420p", yuv_dec_up
                ]
                subprocess.run(decode_up_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                # --- 码率与质量 ---
                video_size_kb = parse_ffmpeg_log_for_video_size(log_path)
                duration_sec = clip_len / float(fps)
                bitrate_kbps = (video_size_kb * 8) / duration_sec # 得到准确 kbps

                dec_clip_bgr_up = read_yuv420_clip_to_bgr(yuv_dec_up, clip_len, 1080, 1920)
                # compute_quality_metric 内部会使用全局的 AlexNet
                V_val = compute_quality_metric(ref_clip_bgr, dec_clip_bgr_up)

                stats[str(qp)]["R"].append(bitrate_kbps)
                stats[str(qp)]["V"].append(V_val)

                # 清理
                for p in [bitstream_path, log_path, yuv_dec_up]:
                    if os.path.exists(p): os.remove(p)

            # 打印当前 Sample 的平均表现，方便观察波动
            sample_r = np.mean([stats[str(qp)]["R"][-1] for qp in qp_list])
            sample_v = np.mean([stats[str(qp)]["V"][-1] for qp in qp_list])
            print(f"Avg_R: {sample_r:7.1f} | Avg_LPIPS: {abs(sample_v):.4f}")

            if os.path.exists(input_yuv_path): os.remove(input_yuv_path)

        # 汇总
        rd_dict[vid] = {}
        print(f"\nFinal RD for {vid}:")
        for qp in qp_list:
            avg_r = float(np.mean(stats[str(qp)]["R"]))
            avg_v = float(np.mean(stats[str(qp)]["V"]))
            rd_dict[vid][str(qp)] = {"R": avg_r, "V": avg_v}
            print(f"  QP {qp}: Bitrate = {avg_r:8.2f} kbps, LPIPS = {abs(avg_v):.4f}")

    with open(output_json_path, "w") as f:
        json.dump(rd_dict, f, indent=2)
    print(f"\n[Align Check] Baseline saved to {output_json_path}")
    
def build_cas_baseline_rd_json_qp_clips(
        video_meta_list: List[Dict],
        output_json_path: str,
        cas_strength: float = 0.5,        # CAS 锐化强度
        samples_per_video: int = 20,     # 每个视频采样多少个 Clip
        clip_len: int = 16,              # 必须与训练时的 clip_len 严格一致
        qp_list: List[int] = None,
        fps: int = 120,
        downscale: int = 2,
        ffmpeg_bin: str = "ffmpeg",
        work_dir: str = "ElegantRL-master/my_RL/rd_cas_baseline_tmp"):
    """
    基于“随机 Clip 采样”的 CAS Baseline 构建函数：
    
    流程：
    对于每个视频 vid：
      1) 循环采样 samples_per_video 次：
         a. 随机读取 16 帧 (1080p) 并下采样到 540p
         b. 编码：540p YUV -> [CAS 滤镜锐化] -> [HEVC 编码]
         c. 计算该 Clip 的实际码率 (包含锐化带来的比特开销)
         d. 解码并上采样回 1080p，计算质量指标 (LPIPS/V)
      2) 汇总所有采样点的 R 和 V 取平均值，保存 JSON
    """
    qp_list = qp_list or [37, 40, 42, 45, 47]
    
    # 清理 / 创建工作目录
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    rd_dict: Dict[str, Dict[str, Dict[str, float]]] = {}

    print("========== [Baseline Build - CAS In-line Sampling] ==========")
    print(f"CAS Strength: {cas_strength}")
    print(f"QP list: {qp_list}")
    print(f"Samples per video: {samples_per_video}")

    for meta in video_meta_list:
        vid = meta["id"]
        W, H = int(meta["width"]), int(meta["height"])
        H_down, W_down = H // downscale, W // downscale

        print(f"\n>>> Processing Video: {vid} (sampling {samples_per_video} clips with CAS)")

        # 初始化统计容器
        stats = {str(qp): {"R": [], "V": []} for qp in qp_list}

        for i in range(samples_per_video):
            # 1) 随机读取一个 Clip (16帧, 1080p YUV)
            y_1080, cb_1080, cr_1080 = load_clip_yuv_planes(meta, clip_len)

            # 2) 构造 1080p BGR 参考 (用于质量评估)
            ref_clip_bgr = yuv420_planes_to_bgr_clip(y_1080, cb_1080, cr_1080)

            # 3) 下采样到编码前分辨率 (540p)
            y_ds, cb_ds, cr_ds = downsample_yuv420_clip(
                y_1080, cb_1080, cr_1080, scale=downscale
            )
            
            # 准备输入文件 (未经处理的 540p YUV)
            yuv_bytes_input = ycbcr420_clip_to_yuv420_bytes(y_ds, cb_ds, cr_ds)
            input_yuv_path = os.path.join(work_dir, f"{vid}_sample_{i}_in.yuv")
            with open(input_yuv_path, "wb") as f:
                f.write(yuv_bytes_input)

            # 4) 遍历所有 QP 进行编码测试
            for qp in qp_list:
                bitstream_path = os.path.join(work_dir, f"{vid}_s{i}_q{qp}.hevc")
                dec_yuv_path = os.path.join(work_dir, f"{vid}_s{i}_q{qp}_dec.yuv")
                log_path = os.path.join(work_dir, f"{vid}_s{i}_q{qp}.log")

                # --- 编码 (内联 CAS 滤镜) ---
                # 逻辑：读取 540p YUV -> 应用 CAS -> 喂给 NVENC 编码器
                encode_cmd = [
                    ffmpeg_bin, "-y",
                    "-f", "rawvideo", "-pix_fmt", "yuv420p",
                    "-s:v", f"{W_down}x{H_down}",
                    "-r", str(fps),
                    "-i", input_yuv_path,
                    
                    # 应用 CAS 滤镜 (关键位置)
                    "-vf", f"cas=strength={cas_strength}",
                    
                    "-c:v", "hevc_nvenc",
                    "-rc", "constqp", "-qp", str(qp),
                    "-preset", "p4",
                    "-profile:v", "main",
                    # "-bf", "3",
                    # "-rc-lookahead", "32",
                    # "-spatial-aq", "1",
                    # "-aq-strength", "8",
                    "-pix_fmt", "yuv420p",
                    bitstream_path,
                ]
                with open(log_path, "w") as f_log:
                    subprocess.run(encode_cmd, check=True,
                                   stdout=subprocess.DEVNULL, stderr=f_log)

                # --- 解析码率 (时间基准是 clip_len) ---
                video_size_kb = parse_ffmpeg_log_for_video_size(log_path)
                duration_sec = clip_len / float(fps)

                if video_size_kb > 0:
                    total_bits = video_size_kb * 1024 * 8
                    bitrate_kbps = total_bits / 1000.0 / duration_sec
                else:
                    bitrate_kbps = os.path.getsize(bitstream_path) * 8.0 / 1000.0 / duration_sec

                # --- 解码 & 上采样 & 计算质量 ---
                decode_cmd = [
                    ffmpeg_bin, "-y",
                    "-i", bitstream_path,
                    "-f", "rawvideo", "-pix_fmt", "yuv420p",
                    dec_yuv_path,
                ]
                subprocess.run(decode_cmd, check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                y_dec, cb_dec, cr_dec = read_yuv420_clip_to_planes(
                    dec_yuv_path, clip_len, H_down, W_down
                )
                y_up, cb_up, cr_up = upsample_yuv420_clip(
                    y_dec, cb_dec, cr_dec, scale=downscale
                )
                dec_clip_bgr = yuv420_planes_to_bgr_clip(y_up, cb_up, cr_up)

                # 计算质量指标
                V_val = compute_quality_metric(ref_clip_bgr, dec_clip_bgr)

                # 记录结果
                stats[str(qp)]["R"].append(bitrate_kbps)
                stats[str(qp)]["V"].append(V_val)

                # 清理临时编码文件
                for p in [bitstream_path, dec_yuv_path, log_path]:
                    if os.path.exists(p): os.remove(p)

            # 进度打印
            if (i + 1) % 5 == 0:
                print(f"    Sample {i+1}/{samples_per_video} done.")
            
            # 清理本轮采样的 YUV
            if os.path.exists(input_yuv_path): os.remove(input_yuv_path)

        # 5) 汇总该视频的所有样本取平均
        rd_dict[vid] = {}
        for qp in qp_list:
            avg_r = float(np.mean(stats[str(qp)]["R"]))
            avg_v = float(np.mean(stats[str(qp)]["V"]))
            rd_dict[vid][str(qp)] = {"R": avg_r, "V": avg_v}
            print(f"  -> QP {qp} (CAS): Avg R={avg_r:.2f}, Avg V={avg_v:.4f}")

    # 6) 保存最终 JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(rd_dict, f, indent=2, ensure_ascii=False)

    print(f"\n[CAS-Baseline] Saved to {output_json_path}")

# ================================================================
# 训练主循环
# ================================================================

def  train_rapnet_with_config(
        video_meta_list: List[Dict],
        rd_json_path: str,
        num_iterations: int = None,
        clip_len: int = None,
        width: int = None,
        height: int = None,
        downscale: int = None,
        ffmpeg_bin: str = None):
    """
    简化版本的训练主循环，每个 iteration:
      1) env.reset() 随机取一个视频 clip；
      2) Actor 给出预处理帧（动作）；
      3) env.step() 在 5 个 QP 上编码，返回 reward；
      4) 用当前样本 (S,A,R) 更新一次 Actor+Critic。
    """

    # ---------- 使用 CONFIG 中的默认值（严格遵循论文设定） ----------
    if num_iterations is None:
        num_iterations = CONFIG["num_iterations"]
    if clip_len is None:
        clip_len = CONFIG["clip_len"]
    if width is None:
        width = CONFIG["width"]
    if height is None:
        height = CONFIG["height"]
    if downscale is None:
        downscale = CONFIG["downscale"]
    if ffmpeg_bin is None:
        ffmpeg_bin = CONFIG["ffmpeg_bin"]

    device_actor, device_critic = resolve_devices()
    H_down = height // downscale
    W_down = width // downscale

    os.makedirs(CONFIG["model_dir"], exist_ok=True)
    writer = SummaryWriter(log_dir=CONFIG["log_dir"])

    # ---------- 创建环境 ----------
    env = NvencClipEnvQP(
        video_meta_list=video_meta_list,
        rd_curve_json=rd_json_path,
        work_dir=CONFIG["work_dir"],
        qp_list=CONFIG["qp_list"],
        clip_len=clip_len,
        fps=CONFIG["fps"],
        width=width,
        height=height,
        downscale=downscale,
        ffmpeg_bin=ffmpeg_bin,
        alpha_q=CONFIG["alpha_q"],
        beta_r=CONFIG["beta_r"],
    )

    # ---------- 创建 AC Agent ----------
    agent = DeterministicACAgentYCbCr(
        H_down=H_down,
        W_down=W_down,
        clip_len=clip_len,
        lr_actor=CONFIG["lr_actor"],
        lr_critic=CONFIG["lr_critic"],
        lambda_pixel=0.0005,
        device_actor=device_actor,
        device_critic=device_critic,
    )

#     # ---------- 尝试加载checkpoint（热启动）----------
#     # 优先加载best权重，如果没有则加载最新的iter权重
# # ---------- 尝试加载指定 checkpoint (1400 step) ----------
#     model_dir ="/workspace/RL/ElegantRL-master/my_RL/checkpoints/rapnet_hevc_12_18_v13"
#     target_iter = 1400  # 设定目标步数
    
#     print(f"\n{'='*70}")
#     print(f"[LOADING SPECIFIC CHECKPOINT: ITER {target_iter}]")
#     print(f"{'='*70}")
    
#     if os.path.exists(model_dir):
#         # 构建指定步数的文件名
#         actor_path = os.path.join(model_dir, f"actor_iter00{target_iter}.pth")
#         critic_path = os.path.join(model_dir, f"critic_iter00{target_iter}.pth")
        
#         # 检查文件是否存在
#         if os.path.exists(actor_path) and os.path.exists(critic_path):
#             try:
#                 # 执行加载
#                 agent.actor.load_state_dict(torch.load(actor_path, map_location=device_actor))
#                 agent.critic.load_state_dict(torch.load(critic_path, map_location=device_critic))
                
#                 # 同步 actor 使用的 critic 副本（如果你的逻辑需要）
#                 if hasattr(agent, 'critic_for_actor'):
#                     agent.critic_for_actor.load_state_dict(agent.critic.state_dict())
                
#                 print(f"✅ [SUCCESS] 已成功加载第 {target_iter} 步的权重")
#                 print(f"   Actor:  {os.path.basename(actor_path)}")
#                 print(f"   Critic: {os.path.basename(critic_path)}")
#             except Exception as e:
#                 print(f"❌ [ERROR] 加载权重时出错: {e}")
#         else:
#             print(f"❌ [ERROR] 未找到指定的权重文件!")
#             print(f"   预期路径: {actor_path}")
#             print(f"   请检查 model_dir 中是否存在该文件。")
#     else:
#         print(f"⚠️  [ERROR] 模型目录不存在: {model_dir}")
    
#     print(f"{'='*70}\n")

    # ---------- 学习率衰减（严格按照论文：每 1000 iters ×1/3） ----------
    scheduler_actor = StepLR(
        agent.optim_actor,
        step_size=CONFIG["lr_step_size"],
        gamma=CONFIG["lr_gamma"],
    )
    scheduler_critic = StepLR(
        agent.optim_critic,
        step_size=CONFIG["lr_step_size"],
        gamma=CONFIG["lr_gamma"],
    )

    best_eval_reward = -1e9

    # ---------- 精确断点续训：从checkpoint读取最后的iter号 ----------
    # last_iter = get_last_iter_simple(CONFIG["log_dir"], CONFIG["/workspace/RL/ElegantRL-master/my_RL/checkpoints/rapnet_hevc_12_18_v13"])
    # start_iter = last_iter + 1
    start_iter =1
    # if last_iter > 0:
    #     print(f"\n{'='*70}")
    #     print(f"🔄 [CHECKPOINT LOADED] 检测到之前训练到iter {last_iter}")
    #     print(f"   ➜ 将从iter {start_iter}继续训练")
    #     print(f"   ➜ 新增的scalar数据会追加到同一log目录中")
    #     print(f"{'='*70}\n")
    # else:
    #     print(f"\n{'='*70}")
    #     print(f"🆕 [NEW TRAINING] 从iter 1开始新训练")
    #     print(f"{'='*70}\n")

    for it in range(start_iter, num_iterations + 1):
        # 【v18】 显式切换 train 模式，启用 act 中的噪声
        agent.actor.train()
        
        # 1) 采样一个 clip
        y_s, cb_s, cr_s, info = env.reset()

        # 2) 当前策略给出预处理后的 clip（动作）
        y_a, cb_a, cr_a = agent.act(y_s, cb_s, cr_s)

        # 3) 与环境交互，得到 reward
        _, reward, done, info_step = env.step(y_a, cb_a, cr_a)

        # 4) 用单步样本更新 Actor + Critic
        loss_actor, loss_critic = agent.update(
            y_state=y_s,
            cb_state=cb_s,
            cr_state=cr_s,
            y_action=y_a,
            cb_action=cb_a,
            cr_action=cr_a,
            reward=reward,
        )

        # 5) 学习率 scheduler 按 iteration 更新
        scheduler_actor.step()
        scheduler_critic.step()
        current_lr = scheduler_actor.get_last_lr()[0]

        # ---------- 日志 & TensorBoard ----------
        writer.add_scalar("focus/reward", reward, it)
        writer.add_scalar("train/loss_actor", loss_actor, it)
        writer.add_scalar("train/loss_critic", loss_critic, it)
        writer.add_scalar("train/lr", current_lr, it)
        # --- 【新增】必须监控的动力学指标 ---
        if hasattr(agent, "last_force_q"):
            # 1. 进攻的力量 (Critic 推力)
            writer.add_scalar("focus/force_q_thrust", agent.last_force_q, it)
            # 2. 防守的力量 (MSE 阻力)
            writer.add_scalar("focus/force_mse_drag", agent.last_force_mse, it)
            # 3. 战场态势 (推阻比) - 最重要！
            # > 1.0 : 进攻方占优，模型正在为了高分改变图像
            # < 1.0 : 防守方占优，模型被 MSE 锁死，倾向于不动
            writer.add_scalar("focus/ratio_force", agent.last_force_ratio, it)
        if "qp_list" in info_step and "R_cur" in info_step:
            # --- per-QP stats ---
            qp_list = info_step.get("qp_list", [])
            for i, qp in enumerate(qp_list):
                if i < len(info_step.get("R_cur", [])):
                    writer.add_scalar(f"train/R_cur_qp{qp}", info_step["R_cur"][i], it)
                if i < len(info_step.get("V_cur", [])):
                    writer.add_scalar(f"train/V_cur_qp{qp}", info_step["V_cur"][i], it)
                if i < len(info_step.get("R_ref", [])):
                    writer.add_scalar(f"train/R_ref_qp{qp}", info_step["R_ref"][i], it)
                if i < len(info_step.get("V_ref", [])):
                    writer.add_scalar(f"train/V_ref_qp{qp}", info_step["V_ref"][i], it)
                if i < len(info_step.get("delta_v", [])):
                    writer.add_scalar(f"train/delta_v_qp{qp}", info_step["delta_v"][i], it)
                if i < len(info_step.get("delta_r", [])):
                    writer.add_scalar(f"train/delta_r_qp{qp}", info_step["delta_r"][i], it)
                if i < len(info_step.get("reward_qp", [])):
                    writer.add_scalar(f"train/reward_qp{qp}", info_step["reward_qp"][i], it)

            # --- mean stats (over QPs) ---
            if len(info_step.get("delta_v", [])) > 0:
                writer.add_scalar("focus/delta_v_mean", float(np.mean(info_step["delta_v"])), it)
            if len(info_step.get("delta_r", [])) > 0:
                writer.add_scalar("focus/delta_r_mean", float(np.mean(info_step["delta_r"])), it)


        y_mean, y_std, y_dmean, y_dl2, y_sat0, y_sat1 = _stats(y_a, y_s)
        cb_mean, cb_std, cb_dmean, cb_dl2, cb_sat0, cb_sat1 = _stats(cb_a, cb_s)
        cr_mean, cr_std, cr_dmean, cr_dl2, cr_sat0, cr_sat1 = _stats(cr_a, cr_s)

        writer.add_scalar("act/Y/mean", y_mean, it)
        writer.add_scalar("act/Y/std", y_std, it)
        writer.add_scalar("act/Y/delta_mean", y_dmean, it)
        writer.add_scalar("act/Y/delta_l2", y_dl2, it)
        writer.add_scalar("act/Y/sat0", y_sat0, it)
        writer.add_scalar("act/Y/sat1", y_sat1, it)

        writer.add_scalar("act/Cb/mean", cb_mean, it)
        writer.add_scalar("act/Cb/std", cb_std, it)
        writer.add_scalar("act/Cb/delta_mean", cb_dmean, it)
        writer.add_scalar("act/Cb/delta_l2", cb_dl2, it)
        writer.add_scalar("act/Cb/sat0", cb_sat0, it)
        writer.add_scalar("act/Cb/sat1", cb_sat1, it)

        writer.add_scalar("act/Cr/mean", cr_mean, it)
        writer.add_scalar("act/Cr/std", cr_std, it)
        writer.add_scalar("act/Cr/delta_mean", cr_dmean, it)
        writer.add_scalar("act/Cr/delta_l2", cr_dl2, it)
        writer.add_scalar("act/Cr/sat0", cr_sat0, it)
        writer.add_scalar("act/Cr/sat1", cr_sat1, it)

        # ---- debug: 纹理/去伪影诊断指标（用 torch 计算，不影响训练）----
        # 注意：这里只是算指标，device 用 actor 的 device 就行
        dev = device_actor

        y_s_t = _to_torch_btchw(y_s, dev)   # (BT,1,H,W)
        y_a_t = _to_torch_btchw(y_a, dev)   # (BT,1,H,W)

        delta_y_t = (y_a_t - y_s_t)

        # 1) 是否还在玩均值/低频：平均绝对改变量（你要的 delta_y_mean_abs）
        delta_y_mean_abs = torch.mean(torch.abs(delta_y_t))

        # 2) 改动的高频占比：HighFreqRatio
        hf_ratio = high_freq_ratio_01(delta_y_t)   # 这里 high_freq_ratio 接受 (BT,1,H,W) 最方便
        hf_out = high_freq_ratio_01(y_a_t)   # 这里 high_freq_ratio 接受 (BT,1,H,W) 最方便
        # 3) 改动是否集中在边缘：EdgeWeightedDelta
        edge_ratio = edge_weighted_delta(y_s_t, y_a_t)  # 这里 edge_weighted_delta 接受 (BT,1,H,W)

        writer.add_scalar("focus/high_freq_delta_ratio", float(hf_ratio.detach().cpu().item()), it)
        writer.add_scalar("debug/high_freq_y_ratio", float(hf_out.detach().cpu().item()), it)
        writer.add_scalar("focus/edge_weighted_delta", float(edge_ratio.detach().cpu().item()), it)
        writer.add_scalar("debug/delta_y_mean_abs", float(delta_y_mean_abs.detach().cpu().item()), it)
        writer.add_scalar("focus/tv_loss", agent.last_tv_loss, it)
        writer.add_scalar("focus/id_loss", agent.last_id_loss, it)
        writer.add_scalar("focus/edge_focus", agent.last_edge_focus, it)
        writer.add_scalar("focus/hf_loss", agent.last_hf_loss, it)
        writer.add_scalar("focus/low_loss", agent.last_low_loss, it)
        writer.add_scalar("focus/mse_loss", agent.last_mse_loss, it)
        
        
        # --- Q / TD error ---
        if hasattr(agent, "last_q_pred"):
            writer.add_scalar("train/q_pred", agent.last_q_pred, it)
        if hasattr(agent, "last_q_actor"):
            writer.add_scalar("train/q_actor", agent.last_q_actor, it)
        if hasattr(agent, "last_td_error"):
            writer.add_scalar("train/td_error_abs", agent.last_td_error, it)

        

        if hasattr(agent, "last_q_actor") and hasattr(agent, "last_pixel_loss_total"):
            writer.add_scalar("train/q_over_pixel", float((-agent.last_q_actor) / (agent.last_pixel_loss_total + 1e-8)), it)
        
        #debug dq_da_norm
        if hasattr(agent, "last_dq_da_norm"):
                    writer.add_scalar("debug/dQ_da_Y_norm", agent.last_dq_da_norm, it)
                    
        # 判据：如果这个值长期 < 1e-5，说明 Critic 彻底“死”了（梯度传不回来）
        # 如果这个值 > 0.01，说明 Critic 很有活力
        if it % CONFIG["log_interval"] == 0:
            print(
                f"[Iter {it:05d}] "
                f"R={reward:.4f}, "
                f"L_actor={loss_actor:.6f}, "
                f"L_critic={loss_critic:.6f}, "
                f"lr={current_lr:.2e}"
            )


    # ---------- 梯度诊断 & TensorBoard ----------
        if it % CONFIG["grad_check_interval"] == 0:
            # 1) 打印诊断（每 50 次打印一次）
            diagnose_ac_training(
                actor=agent.actor,
                critic=agent.critic,
                critic_for_actor=agent.critic_for_actor,
                loss_actor=agent.last_loss_actor,
                loss_critic=agent.last_loss_critic,
            )

            # 2) 写入 TensorBoard：actor/critic 梯度统计
            a = grad_stats(agent.actor)
            c = grad_stats(agent.critic)

            for k, v in a.items():
                writer.add_scalar(f"grad/actor/{k}", v, it)
            for k, v in c.items():
                writer.add_scalar(f"grad/critic/{k}", v, it)

            # 3) 额外写：critic_for_actor 是否真的冻结（trainable param 数）
            trainable = sum(p.requires_grad for p in agent.critic_for_actor.parameters())
            total = sum(1 for _ in agent.critic_for_actor.parameters())
            writer.add_scalar("grad/critic_for_actor/trainable_params", trainable, it)
            writer.add_scalar("grad/critic_for_actor/total_params", total, it)
            
        if it % CONFIG["image_interval"] == 0:
            import matplotlib.pyplot as plt
            from matplotlib.colors import TwoSlopeNorm

            try:
                # 1. 提取首帧并计算对数频谱
                y_s0, y_a0 = y_s[0, 0], y_a[0, 0]
                def get_log_spec(img):
                    fshift = np.fft.fftshift(np.fft.fft2(img))
                    # 归一化幅度，防止图片尺寸影响量级
                    mag = np.abs(fshift) / (img.shape[0] * img.shape[1])
                    return np.log(mag + 1e-6)

                spec_diff = get_log_spec(y_a0) - get_log_spec(y_s0)

                # 2. 动态计算阈值，确保即使微小改动也能“拉开”颜色
                # 使用 abs 的 99.9% 分位数，过滤掉个别极值点噪声，使整体对比度更好
                v_limit = np.percentile(np.abs(spec_diff), 99.9)
                v_limit = max(v_limit, 1e-5) # 保证不为0

                fig, ax = plt.subplots(figsize=(8, 6))
                v_limit_fixed = 0.05 
                norm = TwoSlopeNorm(vmin=-v_limit_fixed, vcenter=0, vmax=v_limit_fixed)
                
                # 使用 interpolation='bilinear' 可以让高频噪声看起来更像云团而非碎点，易于观察趋势
                im = ax.imshow(spec_diff, cmap='seismic', norm=norm, interpolation='bilinear')
                
                # 画出十字辅助线，方便观察论文提到的“轴向增强”
                h, w = spec_diff.shape
                ax.axhline(h//2, color='black', lw=0.5, alpha=0.3)
                ax.axvline(w//2, color='black', lw=0.5, alpha=0.3)
                
                # 获取当前视频名称
                current_vid = info.get("video_id", "Unknown")
                
                # 修改标题显示视频名称
                ax.set_title(f"Spectral Log-Diff (Ratio)\nIter: {it} (Video: {current_vid})")
                ax.axis('off')
                fig.colorbar(im, ax=ax, label='Log Energy Ratio (Action/State)')

                # 3. 写入 TensorBoard
                fig.canvas.draw()
                image_np = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                w, h = fig.canvas.get_width_height()
                image_np = image_np.reshape((h, w, 4))
                image_rgb = image_np[:, :, :3].transpose(2, 0, 1)
                writer.add_image("img/Y/spectral_analysis_adaptive", image_rgb, it)
                writer.add_scalar("focus/spectral_v_limit", v_limit, it)
                plt.close(fig)

            except Exception as e:
                print(f"[Warn] Spectral analysis failed: {e}")
            # 取第0帧、0通道（你的 shape 是 (T,1,H,W)）
            y0  = y_s[0, 0]
            ya0 = y_a[0, 0]
            yd0 = np.abs(ya0 - y0)
            # 对 diff_x100 进行轻微高斯模糊，去除高频噪声，避免 TensorBoard 缩放时的棋盘格伪影
            diff_vis_raw = np.clip(np.abs(ya0 - y0) * 100.0, 0, 1)
            diff_vis = cv2.GaussianBlur(diff_vis_raw, (3, 3), sigmaX=0.5, sigmaY=0.5)

            writer.add_image("img/Y/state",  y0,  it, dataformats="HW")
            writer.add_image("img/Y/action", ya0, it, dataformats="HW")
            writer.add_image("img/Y/diff",   yd0, it, dataformats="HW")
            writer.add_image("img/Y/diff_x100", diff_vis, it, dataformats="HW")
            
            
            cb0  = cb_s[0, 0]
            cba0 = cb_a[0, 0]
            cbd0 = np.abs(cba0 - cb0)
            diff_vis_cb_raw = np.clip(np.abs(cba0 - cb0) * 100.0, 0, 1)
            diff_vis_cb = cv2.GaussianBlur(diff_vis_cb_raw, (3, 3), sigmaX=0.5, sigmaY=0.5)
            cr0  = cr_s[0, 0]
            cra0 = cr_a[0, 0]
            crd0 = np.abs(cra0 - cr0)
            diff_vis_cr_raw = np.clip(np.abs(cra0 - cr0) * 100.0, 0, 1)
            diff_vis_cr = cv2.GaussianBlur(diff_vis_cr_raw, (3, 3), sigmaX=0.5, sigmaY=0.5)
            writer.add_image("img/Cb/state",  cb0,  it, dataformats="HW")
            writer.add_image("img/Cb/action", cba0, it, dataformats="HW")
            writer.add_image("img/Cb/diff",   cbd0, it, dataformats="HW")
            writer.add_image("img/Cb/diff_x100", diff_vis_cb, it, dataformats="HW")

            writer.add_image("img/Cr/state",  cr0,  it, dataformats="HW")
            writer.add_image("img/Cr/action", cra0, it, dataformats="HW")
            writer.add_image("img/Cr/diff",   crd0, it, dataformats="HW")
            writer.add_image("img/Cr/diff_x100", diff_vis_cr, it, dataformats="HW")   


  
        # ---------- 周期性验证 ----------
        if it % CONFIG["eval_interval"] == 0:
            eval_reward = evaluate_agent_once(env, agent, num_clips=3)
            writer.add_scalar("eval/avg_reward", eval_reward, it)
            print(f"[Eval] Iter {it:05d}, Fixed Avg Reward={eval_reward:.4f}")

            # 保存 best model（基于 eval reward）
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                best_actor_path = os.path.join(CONFIG["model_dir"], "actor_best.pth")
                best_critic_path = os.path.join(CONFIG["model_dir"], "critic_best.pth")
                torch.save(agent.actor.state_dict(), best_actor_path)
                torch.save(agent.critic.state_dict(), best_critic_path)
                print(f"[Model] New best model saved to {best_actor_path}")

        # ---------- 周期性 checkpoint ----------
        if it % CONFIG["save_interval"] == 0:
            ckpt_actor = os.path.join(CONFIG["model_dir"], f"actor_iter{it:06d}.pth")
            ckpt_critic = os.path.join(CONFIG["model_dir"], f"critic_iter{it:06d}.pth")
            torch.save(agent.actor.state_dict(), ckpt_actor)
            torch.save(agent.critic.state_dict(), ckpt_critic)
            print(f"[Checkpoint] Saved at iter {it}")

    writer.close()

#-----------------------DEBUG-----------------------
@torch.no_grad()
def evaluate_agent_once(env: NvencClipEnvQP,
                        agent: DeterministicACAgentYCbCr,
                        num_clips: int = 3) -> float:
    """固定对 3 个特定视频进行验证，以消除随机性。"""
    agent.actor.eval()
    agent.critic.eval()
    
    import random
    # 保存当前的随机状态，避免影响训练的数据采样
    old_state = random.getstate()
    
    # 固定种子，确保每次验证时选取的 16 帧图像完全相同
    random.seed(42)
    np.random.seed(42)
    
    # 严格固定这三个 ID
    target_ids = ["shakendry_1080p", "bosphorus_1080p", "honeybee_1080p"]
    eval_metas = [m for m in env.video_meta_list if m["id"] in target_ids]
    
    rewards = []
    for meta in eval_metas:
        # 1. 重置环境到特定视频 (这会填充 env.ref_clip_1080p 和相关状态)
        env.current_video_id = meta["id"]
        y_1080, cb_1080, cr_1080 = load_clip_yuv_planes(meta, env.clip_len)
        env.ref_clip_1080p = yuv420_planes_to_bgr_clip(y_1080, cb_1080, cr_1080)
        y_ds, cb_ds, cr_ds = downsample_yuv420_clip(y_1080, cb_1080, cr_1080, scale=env.downscale)
        env.state_y, env.state_cb, env.state_cr = y_ds, cb_ds, cr_ds

        # 2. Actor 执行动作
        y_a, cb_a, cr_a = agent.act(y_ds, cb_ds, cr_ds)
        
        # 3. Step (这里会自动计算该片段在所有 QP 下的平均奖励)
        _, reward, _, _ = env.step(y_a, cb_a, cr_a)
        rewards.append(reward)
    
    # 恢复训练的随机性状态
    random.setstate(old_state) 
    
    # 返回这 3 个片段的平均奖励
    return float(np.mean(rewards)) if rewards else 0.0
    
    


VIDEO_META_LIST = [
    {
        "id": "beauty_1080p",
        "path": "/datasets/UVG/original_yuv/Beauty_1920x1080_120fps_420_8bit_YUV.yuv",
        "width": 1920,
        "height": 1080,
        "fps": 120,
    },
    {
        "id": "bosphorus_1080p",
        "path": "/datasets/UVG/original_yuv/Bosphorus_1920x1080_120fps_420_8bit_YUV.yuv",
        "width": 1920,
        "height": 1080,
        "fps": 120,
    },
    {
        "id": "honeybee_1080p",
        "path": "/datasets/UVG/original_yuv/HoneyBee_1920x1080_120fps_420_8bit_YUV.yuv",
        "width": 1920,
        "height": 1080,
        "fps": 120,
    },
    {
        "id": "jockey_1080p",
        "path": "/datasets/UVG/original_yuv/Jockey_1920x1080_120fps_420_8bit_YUV.yuv",
        "width": 1920,
        "height": 1080,
        "fps": 120,
    },
    {
        "id": "readysteadygo_1080p",
        "path": "/datasets/UVG/original_yuv/ReadySteadyGo_1920x1080_120fps_420_8bit_YUV.yuv",
        "width": 1920,
        "height": 1080,
    },
    {
        "id": "shakendry_1080p",
        "path": "/datasets/UVG/original_yuv/ShakeNDry_1920x1080_120fps_420_8bit_YUV.yuv",
        "width": 1920,
        "height": 1080,
        "fps": 120,
    },
    {
        "id": "basketballdrive_1080p",
        "path": "/datasets/HEVC_test_sequences/ClassB/BasketballDrive_1920x1080_50.yuv",
        "width": 1920,
        "height": 1080,
        "fps": 50,
    },
    {
        "id": "parkscene_1080p",
        "path": "/datasets/HEVC_test_sequences/ClassB/ParkScene_1920x1080_24.yuv",
        "width": 1920,
        "height": 1080,
        "fps": 24,
    },
    {
        "id": "kimono1_1080p",
        "path": "/datasets/HEVC_test_sequences/ClassB/Kimono1_1920x1080_24.yuv",
        "width": 1920,
        "height": 1080,
        "fps": 24,
    },
    {
        "id": "cactus_1080p",
        "path": "/datasets/HEVC_test_sequences/ClassB/Cactus_1920x1080_50.yuv",
        "width": 1920,
        "height": 1080,
        "fps": 50,
    },
    {
        "id": "bqterrace_1080p",
        "path": "/datasets/HEVC_test_sequences/ClassB/BQTerrace_1920x1080_60.yuv",
        "width": 1920,
        "height": 1080,
        "fps": 60,
    },
    {
        "id": "yachtride_1080p",
        "path": "/datasets/UVG/original_yuv/YachtRide_1920x1080_120fps_420_8bit_YUV.yuv",
        "width": 1920,
        "height": 1080,
    }
]

if __name__ == "__main__":
    DEBUG_SHAPES = False  
    # debug_one_step_multi_gpu()

    video_meta_list = VIDEO_META_LIST
    # build_baseline_rd_json_qp_clips(
    #     video_meta_list,
    #     output_json_path="ElegantRL-master/my_RL/result/rd_baseline_hevc_qp_full_clip_lowbits.json",
    #     qp_list=CONFIG["qp_list"],
    #     fps=CONFIG["fps"],
    #     downscale=CONFIG["downscale"],
    #     ffmpeg_bin=CONFIG["ffmpeg_bin"],       
        
    # build_baseline_rd_json_qp_fullvideo(
    #     video_meta_list,
    #     output_json_path="/workspace/RL/ElegantRL-master/my_RL/result/rd_baseline_hevc_test_qp_full_alex.json",
    #     qp_list=CONFIG["qp_list"],
    #     fps=CONFIG["fps"],
    #     downscale=CONFIG["downscale"],
    #     ffmpeg_bin=CONFIG["ffmpeg_bin"],
    #  )


    
    train_rapnet_with_config(
        video_meta_list=video_meta_list,
        rd_json_path=os.environ.get(
            "RAPNET_BASELINE_JSON",
            os.path.join(DEFAULT_RESULT_DIR, "rd_baseline_hevc_qp_full_clip_lowbits_alex.json"),
        ),
        # rd_json_path="/workspace/RL/ElegantRL-master/my_RL/result/rd_baseline_hevc_test_qp_full_alex.json",
    )
    

    pass
