
import os
import sys
import json
import torch
import numpy as np
import logging
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

def save_first_frame_spectral_viz(img_s0, img_a0, vid, out_path):
    """
    img_s0/img_a0: 2D array (H, W), 例如 downsample 后的 Y 通道第一帧
    """
    f_s = np.fft.fftshift(np.fft.fft2(img_s0))
    f_a = np.fft.fftshift(np.fft.fft2(img_a0))

    # 归一化幅度（避免不同分辨率/能量导致不可比）
    spec_s = np.abs(f_s) / (img_s0.shape[0] * img_s0.shape[1])
    spec_a = np.abs(f_a) / (img_a0.shape[0] * img_a0.shape[1])

    eps = 1e-3
    spec_diff = (spec_a - spec_s) / (spec_s + eps)

    v_limit = 0.04
    norm = TwoSlopeNorm(vmin=-v_limit, vcenter=0, vmax=v_limit)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(spec_diff, cmap="seismic", norm=norm, interpolation="bilinear")
    ax.set_title(f"Regularized Spectral Change (First Frame): {vid}\n(Red: Enhancement | Blue: Suppression)")
    ax.axis("off")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Regularized Relative Change (Ratio)")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
# 从您的主训练脚本导入组件
from rapnet_hevc_ppo_qp_yuv_simple import (
    RAPNetYCbCr, CONFIG,
    VIDEO_META_LIST,
    read_yuv420_clip_to_planes,
    downsample_yuv420_clip,
    upsample_yuv420_clip,
    yuv420_planes_to_bgr_clip,
    ycbcr420_clip_to_yuv420_bytes,
    compute_quality_metric,
    parse_ffmpeg_log_for_video_size,
    subprocess, tempfile
)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("FullVideoEval")

def evaluate_full_video(actor, video_meta, rd_baseline, device):
    vid = video_meta["id"]
    path = video_meta["path"]
    W, H = video_meta["width"], video_meta["height"]
    fps = int(video_meta.get("fps", CONFIG["fps"]))
    scale = CONFIG["downscale"]
    chunk_size = CONFIG["clip_len"]

    if vid not in rd_baseline:
        logger.error(f"\n[Error] 跳过视频 {vid}：Baseline 中未包含此视频 ID")
        return None

    frame_bytes = int(W * H * 1.5)
    total_frames = os.path.getsize(path) // frame_bytes
    logger.info(f"\n{'='*80}")
    logger.info(f"评估视频: {vid} | 帧数: {total_frames}")
    logger.info(f"{'='*80}")
    
    y_all, cb_all, cr_all = read_yuv420_clip_to_planes(path, total_frames, H, W)
    ref_bgr_all = yuv420_planes_to_bgr_clip(y_all, cb_all, cr_all)

    y_ds, cb_ds, cr_ds = downsample_yuv420_clip(y_all, cb_all, cr_all, scale=scale)
    H_ds, W_ds = y_ds.shape[2], y_ds.shape[3]

    actor.eval()
    y_out_list, cb_out_list, cr_out_list = [], [], []
    
    with torch.no_grad():
        for i in tqdm(range(0, total_frames, chunk_size), desc="Inferring"):
            end = min(i + chunk_size, total_frames)
            y_in = torch.from_numpy(y_ds[i:end]).float().to(device).unsqueeze(0)
            cb_in = torch.from_numpy(cb_ds[i:end]).float().to(device).unsqueeze(0)
            cr_in = torch.from_numpy(cr_ds[i:end]).float().to(device).unsqueeze(0)
            
            y_a, cb_a, cr_a = actor(y_in, cb_in, cr_in)
            y_out_list.append(y_a.squeeze(0).cpu().numpy())
            cb_out_list.append(cb_a.squeeze(0).cpu().numpy())
            cr_out_list.append(cr_a.squeeze(0).cpu().numpy())

    y_action = np.concatenate(y_out_list, axis=0)
    cb_action = np.concatenate(cb_out_list, axis=0)
    cr_action = np.concatenate(cr_out_list, axis=0)
    
    # # --- 每个视频做一次：第一帧频谱可视化（输入vs输出） ---
    # try:
    #     # 注意：y_ds / y_action 的 shape 是 (T, 1, H, W)
    #     img_s0 = y_ds[0, 0]
    #     img_a0 = y_action[0, 0]

    #     out_dir = "/workspace/RL/ElegantRL-master/my_RL/result/full_eval"
    #     os.makedirs(out_dir, exist_ok=True)
    #     out_path = os.path.join(out_dir, f"spectral_{vid}.png")

    #     save_first_frame_spectral_viz(img_s0, img_a0, vid, out_path)
    #     logger.info(f"[Viz] Saved spectral first-frame analysis: {out_path}")
    # except Exception as e:
    #     logger.warning(f"[Viz] Spectral visualization failed for {vid}: {e}")
        
        
    results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        yuv_processed_path = os.path.join(tmpdir, "full_processed.yuv")
        with open(yuv_processed_path, "wb") as f:
            f.write(ycbcr420_clip_to_yuv420_bytes(y_action, cb_action, cr_action))

        for qp in CONFIG["qp_list"]:
            bitstream = os.path.join(tmpdir, f"out_q{qp}.hevc")
            log_path = os.path.join(tmpdir, f"log_q{qp}.txt")
            yuv_dec = os.path.join(tmpdir, f"dec_q{qp}.yuv")

            cmd_enc = [
                CONFIG["ffmpeg_bin"], "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p",
                "-s:v", f"{W_ds}x{H_ds}", "-r", str(fps), "-i", yuv_processed_path,
                "-c:v", "hevc_nvenc", "-rc", "constqp", "-qp", str(qp),
                "-preset", "p4", "-profile:v", "main",  "-bf", "3",
                 "-rc-lookahead", "32",  "-spatial-aq", "1", "-aq-strength", "8",
                "-pix_fmt", "yuv420p", bitstream
            ]
            subprocess.run(cmd_enc, check=True, stderr=open(log_path, "w"), stdout=subprocess.DEVNULL)

            video_size_kb = parse_ffmpeg_log_for_video_size(log_path)
            duration = total_frames / fps
            bitrate_cur = (video_size_kb * 1024 * 8) / 1000.0 / duration 

            subprocess.run([CONFIG["ffmpeg_bin"], "-y", "-i", bitstream, "-f", "rawvideo", "-pix_fmt", "yuv420p", yuv_dec], 
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            y_d, cb_d, cr_d = read_yuv420_clip_to_planes(yuv_dec, total_frames, H_ds, W_ds)
            y_up, cb_up, cr_up = upsample_yuv420_clip(y_d, cb_d, cr_d, scale=scale)
            dec_bgr_all = yuv420_planes_to_bgr_clip(y_up, cb_up, cr_up)
            
            # --- 关键修正：为了与 Baseline JSON 保持一致，LPIPS 取负 ---
            lpips_frames = 60
            n_lpips = min(lpips_frames, total_frames)

            lpips_dist = compute_quality_metric(ref_bgr_all[:n_lpips], dec_bgr_all[:n_lpips])
            v_cur = -lpips_dist

            ref = rd_baseline[vid][str(qp)]
            delta_v = (v_cur - ref["V"]) / (abs(ref["V"]) + 1e-6)
            delta_r = (ref["R"] - bitrate_cur) / (ref["R"] + 1e-6)
            reward = CONFIG["alpha_q"] * delta_v + CONFIG["beta_r"] * delta_r

            results.append({
                "video_id": vid,
                "qp": qp, 
                "R": bitrate_cur, 
                "V": v_cur, 
                "reward": reward, 
                "delta_r": delta_r, 
                "delta_v": delta_v
            })

    return results

def main():
    CKPT = os.environ.get(
        "RAPNET_ACTOR_CKPT",
        os.path.join(ROOT_DIR, "checkpoints", "actor_best.pth"),
    )
    BASE_JSON = os.environ.get(
        "RAPNET_BASELINE_JSON",
        os.path.join(ROOT_DIR, "results", "rd_baseline_hevc_test_qp_full_alex.json"),
    )
    OUTPUT_JSON = os.environ.get(
        "RAPNET_OUTPUT_JSON",
        os.path.join(ROOT_DIR, "results", "eval_full_video_results.json"),
    )
    
    device = torch.device(os.environ.get("RAPNET_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"))
    
    actor = RAPNetYCbCr().to(device)
    if os.path.exists(CKPT):
        actor.load_state_dict(torch.load(CKPT, map_location=device))
        logger.info(f"[*] 成功加载权重: {CKPT}")
    else:
        logger.error(f"[Error] 未找到权重文件: {CKPT}")
        return

    if not os.path.exists(BASE_JSON):
        logger.error(f"[Error] 未找到Baseline JSON: {BASE_JSON}")
        return

    with open(BASE_JSON, "r") as f:
        rd_baseline = json.load(f)

    video_list = VIDEO_META_LIST

    all_results = []
    for video in video_list:
        res = evaluate_full_video(actor, video, rd_baseline, device)
        if res: all_results.extend(res)

    # --- 统计与展示部分 ---
    df = pd.DataFrame(all_results)

    # 1. 打印详细 R-V 统计表 (横轴为 QP，纵轴为视频)
    logger.info("\n" + "="*95 + "\nDETAILED FULL-VIDEO RD STATISTICS (AGENT)\n" + "="*95)
    rv_pivot = df.pivot(index='video_id', columns='qp', values=['R', 'V'])
    print(rv_pivot)

    # 2. 打印奖励统计表
    logger.info("\n" + "="*95 + "\nDETAILED REWARD & DELTA STATISTICS\n" + "="*95)
    reward_pivot = df.pivot(index='video_id', columns='qp', values=['reward', 'delta_r', 'delta_v'])
    print(reward_pivot)

    # 3. 总体 QP 平均统计
    logger.info("\n" + "="*40 + " SUMMARY BY QP " + "="*40)
    print(df.groupby("qp")[["R", "V", "delta_r", "delta_v", "reward"]].mean())

    # 4. 导出为 JSON
    output_dict = {}
    for _, row in df.iterrows():
        vid = row['video_id']
        qp = str(int(row['qp']))
        if vid not in output_dict: output_dict[vid] = {}
        output_dict[vid][qp] = {
            "R": round(float(row['R']), 2),
            "V": round(float(row['V']), 6),
            "reward": round(float(row['reward']), 4)
        }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output_dict, f, indent=2)
    
    logger.info(f"\n[Done] 完整视频评估结果已保存至: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
