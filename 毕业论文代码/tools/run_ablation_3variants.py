#!/usr/bin/env python3
"""
Ablation Study Runner
=====================
Train 3 variants sequentially + unified BD-LPIPS evaluation, all in one run.

Variant 1 (v1_pg_only):  loss = -Q(S,A(S))                                    [w_mse=0,  w_hf=0 ]
Variant 2 (v2_pg_mse):   loss = -Q(S,A(S)) + 5 * loss_mse                     [w_mse=5,  w_hf=0 ]
Variant 3 (v3_full):     loss = -Q(S,A(S)) + 5 * loss_mse + 10 * loss_hf      [w_mse=5,  w_hf=10]

Usage:
    python run_ablation_3variants.py
"""

import os
import sys
import json
import csv
import copy
import time
import datetime
import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ================================================================
# Import all components from base training module (avoid code duplication)
# ================================================================
from rapnet_hevc_ppo_qp_yuv_1080p import (
    CONFIG,
    VIDEO_META_LIST,
    train_rapnet_with_config,
    NvencClipEnvQP,
    DeterministicACAgentYCbCr,
    RAPNetYCbCr,
    load_clip_yuv_planes,
    random_crop_yuv420_clip,
    yuv420_planes_to_bgr_clip,
    ycbcr420_clip_to_yuv420_bytes,
    read_yuv420_clip_to_planes,
    compute_quality_metric,
    parse_ffmpeg_log_for_video_size,
    resolve_devices,
    init_lpips_model,
)

# ================================================================
# Ablation Experiment Config
# ================================================================
NUM_ITERS = 5000           # iterations per variant
QUALITY_METRIC = "lpips"   # quality metric
RD_JSON_PATH = os.environ.get(
    "RAPNET_BASELINE_JSON",
    os.path.join(ROOT_DIR, "results", "rd_baseline_crop30_lpips.json"),
)

# output root directory
OUTPUT_ROOT = os.environ.get("RAPNET_OUTPUT_ROOT", os.path.join(ROOT_DIR, "results"))

VARIANTS = [
    {
        "name": "v1_pg_only",
        "desc": "Policy gradient only (w_mse=0, w_hf=0)",
        "w_mse": 0.0,
        "w_hf":  0.0,
    },
    {
        "name": "v2_pg_mse",
        "desc": "+ MSE regularization (w_mse=5, w_hf=0)",
        "w_mse": 5.0,
        "w_hf":  0.0,
    },
    {
        "name": "v3_full",
        "desc": "Full scheme (w_mse=5, w_hf=10)",
        "w_mse": 5.0,
        "w_hf":  10.0,
    },
]

# Eval video IDs (all training videos used for evaluation)
EVAL_VIDEO_IDS = [
    "beauty_1080p",
    "bosphorus_1080p",
    "honeybee_1080p",
    "jockey_1080p",
    "readysteadygo_1080p",
    "shakendry_1080p",
    "basketballdrive_1080p",
    "parkscene_1080p",
    "kimono1_1080p",
    "cactus_1080p",
    "bqterrace_1080p",
    "yachtride_1080p",
]


# ================================================================
# BD-rate / BD-LPIPS 计算 (Bjøntegaard Delta)
# ================================================================
def bd_rate(R1, Q1, R2, Q2):
    """
    Compute BD-rate (Bjontegaard Delta Rate).

    Args:
        R1, Q1: anchor (baseline) bitrate and quality arrays (sorted by QP)
        R2, Q2: test (variant) bitrate and quality arrays

    Returns:
        BD-rate (%): negative means bitrate saving, positive means increase
    """
    # Need at least 4 points for cubic interpolation
    if len(R1) < 4 or len(R2) < 4:
        return _bd_rate_linear(R1, Q1, R2, Q2)

    lR1 = np.log(np.array(R1, dtype=np.float64))
    lR2 = np.log(np.array(R2, dtype=np.float64))
    Q1 = np.array(Q1, dtype=np.float64)
    Q2 = np.array(Q2, dtype=np.float64)

    # Fit cubic polynomial with Q as independent var, log(R) as dependent var
    p1 = np.polyfit(Q1, lR1, min(3, len(Q1) - 1))
    p2 = np.polyfit(Q2, lR2, min(3, len(Q2) - 1))

    # Integrate over common Q range
    q_min = max(Q1.min(), Q2.min())
    q_max = min(Q1.max(), Q2.max())

    if q_max <= q_min:
        return 0.0

    # 积分 log(R) 关于 Q (using antiderivative of polynomial)
    p1_int = np.polyint(p1)
    p2_int = np.polyint(p2)

    avg1 = (np.polyval(p1_int, q_max) - np.polyval(p1_int, q_min)) / (q_max - q_min)
    avg2 = (np.polyval(p2_int, q_max) - np.polyval(p2_int, q_min)) / (q_max - q_min)

    bd = (np.exp(avg2 - avg1) - 1.0) * 100.0
    return bd


def _bd_rate_linear(R1, Q1, R2, Q2):
    """Linear interpolation fallback when fewer than 4 data points."""
    lR1 = np.log(np.array(R1, dtype=np.float64))
    lR2 = np.log(np.array(R2, dtype=np.float64))
    Q1 = np.array(Q1, dtype=np.float64)
    Q2 = np.array(Q2, dtype=np.float64)

    p1 = np.polyfit(Q1, lR1, min(1, len(Q1) - 1))
    p2 = np.polyfit(Q2, lR2, min(1, len(Q2) - 1))

    q_min = max(Q1.min(), Q2.min())
    q_max = min(Q1.max(), Q2.max())

    if q_max <= q_min:
        return 0.0

    p1_int = np.polyint(p1)
    p2_int = np.polyint(p2)

    avg1 = (np.polyval(p1_int, q_max) - np.polyval(p1_int, q_min)) / (q_max - q_min)
    avg2 = (np.polyval(p2_int, q_max) - np.polyval(p2_int, q_min)) / (q_max - q_min)

    bd = (np.exp(avg2 - avg1) - 1.0) * 100.0
    return bd


def bd_quality(R1, Q1, R2, Q2):
    """
    Compute BD-LPIPS / BD-Quality (Bjontegaard Delta Quality).

    Args:
        R1, Q1: anchor (baseline) bitrate and quality arrays
        R2, Q2: test (variant) bitrate and quality arrays

    Returns:
        BD-Quality: positive means quality improvement
    """
    if len(R1) < 4 or len(R2) < 4:
        return _bd_quality_linear(R1, Q1, R2, Q2)

    lR1 = np.log(np.array(R1, dtype=np.float64))
    lR2 = np.log(np.array(R2, dtype=np.float64))
    Q1 = np.array(Q1, dtype=np.float64)
    Q2 = np.array(Q2, dtype=np.float64)

    # Fit with log(R) as independent var, Q as dependent var
    p1 = np.polyfit(lR1, Q1, min(3, len(lR1) - 1))
    p2 = np.polyfit(lR2, Q2, min(3, len(lR2) - 1))

    r_min = max(lR1.min(), lR2.min())
    r_max = min(lR1.max(), lR2.max())

    if r_max <= r_min:
        return 0.0

    p1_int = np.polyint(p1)
    p2_int = np.polyint(p2)

    avg1 = (np.polyval(p1_int, r_max) - np.polyval(p1_int, r_min)) / (r_max - r_min)
    avg2 = (np.polyval(p2_int, r_max) - np.polyval(p2_int, r_min)) / (r_max - r_min)

    return avg2 - avg1


def _bd_quality_linear(R1, Q1, R2, Q2):
    """Linear interpolation fallback."""
    lR1 = np.log(np.array(R1, dtype=np.float64))
    lR2 = np.log(np.array(R2, dtype=np.float64))
    Q1 = np.array(Q1, dtype=np.float64)
    Q2 = np.array(Q2, dtype=np.float64)

    p1 = np.polyfit(lR1, Q1, min(1, len(lR1) - 1))
    p2 = np.polyfit(lR2, Q2, min(1, len(lR2) - 1))

    r_min = max(lR1.min(), lR2.min())
    r_max = min(lR1.max(), lR2.max())

    if r_max <= r_min:
        return 0.0

    p1_int = np.polyint(p1)
    p2_int = np.polyint(p2)

    avg1 = (np.polyval(p1_int, r_max) - np.polyval(p1_int, r_min)) / (r_max - r_min)
    avg2 = (np.polyval(p2_int, r_max) - np.polyval(p2_int, r_min)) / (r_max - r_min)

    return avg2 - avg1


# ================================================================
# Evaluation: compute LPIPS and bitrate per QP for a given checkpoint
# ================================================================
import tempfile
import subprocess
import random as _random


@torch.no_grad()
def evaluate_variant_rd(
    actor_path: str,
    video_meta_list,
    eval_video_ids,
    qp_list,
    clip_len=16,
    crop_h=540,
    crop_w=960,
    fps=120,
    ffmpeg_bin="ffmpeg",
    num_samples=5,
):
    """
    Load actor_best.pth for a variant, evaluate on multiple clips per test video.

    Returns:
        results: dict  {video_id: {"R": [r_qp1,...], "V": [v_qp1,...]} }
        baseline_results: dict  {video_id: {"R": [...], "V": [...]} }
    """
    device_actor, _ = resolve_devices()
    init_lpips_model()

    # Create Actor and load weights
    actor = RAPNetYCbCr().to(device_actor)
    actor.load_state_dict(torch.load(actor_path, map_location=device_actor))
    actor.eval()

    eval_metas = [m for m in video_meta_list if m["id"] in eval_video_ids]

    results = {}       # variant RD data
    baseline_results = {} # baseline RD data (without Actor)

    work_dir = os.environ.get(
        "RAPNET_TMP_DIR",
        os.path.join(ROOT_DIR, "tmp", "ablation_eval_tmp"),
    )
    os.makedirs(work_dir, exist_ok=True)

    for meta in eval_metas:
        vid = meta["id"]
        print(f"  Evaluating {vid}...")

        # Fix random seed so all variants evaluate on the same clips
        _random.seed(12345)
        np.random.seed(12345)

        # Collect bitrate and quality per QP
        R_variant_per_qp = {qp: [] for qp in qp_list}
        V_variant_per_qp = {qp: [] for qp in qp_list}
        R_baseline_per_qp = {qp: [] for qp in qp_list}
        V_baseline_per_qp = {qp: [] for qp in qp_list}

        for sample_idx in range(num_samples):
            # 1. Load and crop
            y_1080, cb_1080, cr_1080 = load_clip_yuv_planes(meta, clip_len)
            y_crop, cb_crop, cr_crop = random_crop_yuv420_clip(
                y_1080, cb_1080, cr_1080, crop_h, crop_w
            )
            ref_bgr = yuv420_planes_to_bgr_clip(y_crop, cb_crop, cr_crop)

            # 2. Actor inference
            y_t = torch.from_numpy(y_crop).float().to(device_actor).unsqueeze(0)
            cb_t = torch.from_numpy(cb_crop).float().to(device_actor).unsqueeze(0)
            cr_t = torch.from_numpy(cr_crop).float().to(device_actor).unsqueeze(0)
            y_a, cb_a, cr_a = actor(y_t, cb_t, cr_t)
            y_a = torch.clamp(y_a, 0.0, 1.0).squeeze(0).cpu().numpy()
            cb_a = torch.clamp(cb_a, 0.0, 1.0).squeeze(0).cpu().numpy()
            cr_a = torch.clamp(cr_a, 0.0, 1.0).squeeze(0).cpu().numpy()

            T = clip_len
            H_enc, W_enc = crop_h, crop_w

            with tempfile.TemporaryDirectory(dir=work_dir) as tmpdir:
                # --- Variant (preprocessed by Actor) ---
                yuv_variant = os.path.join(tmpdir, "variant.yuv")
                yuv_bytes_v = ycbcr420_clip_to_yuv420_bytes(y_a, cb_a, cr_a)
                with open(yuv_variant, "wb") as f:
                    f.write(yuv_bytes_v)

                # --- Baseline (original, no Actor) ---
                yuv_baseline = os.path.join(tmpdir, "baseline.yuv")
                yuv_bytes_b = ycbcr420_clip_to_yuv420_bytes(y_crop, cb_crop, cr_crop)
                with open(yuv_baseline, "wb") as f:
                    f.write(yuv_bytes_b)

                for qp in qp_list:
                    for tag, yuv_in, R_dict, V_dict in [
                        ("variant", yuv_variant, R_variant_per_qp, V_variant_per_qp),
                        ("baseline", yuv_baseline, R_baseline_per_qp, V_baseline_per_qp),
                    ]:
                        bitstream = os.path.join(tmpdir, f"{tag}_qp{qp}.hevc")
                        yuv_dec = os.path.join(tmpdir, f"{tag}_dec_qp{qp}.yuv")
                        log_path = os.path.join(tmpdir, f"{tag}_enc_qp{qp}.log")

                        # Encode
                        encode_cmd = [
                            ffmpeg_bin, "-y",
                            "-f", "rawvideo", "-pix_fmt", "yuv420p",
                            "-s:v", f"{W_enc}x{H_enc}",
                            "-r", str(fps),
                            "-i", yuv_in,
                            "-c:v", "hevc_nvenc",
                            "-rc", "constqp", "-qp", str(qp),
                            "-preset", "p4", "-profile:v", "main",
                            "-pix_fmt", "yuv420p",
                            bitstream,
                        ]
                        with open(log_path, "w") as f_log:
                            subprocess.run(encode_cmd, check=True,
                                           stdout=subprocess.DEVNULL, stderr=f_log)

                        # Decode
                        decode_cmd = [
                            ffmpeg_bin, "-y",
                            "-i", bitstream,
                            "-f", "rawvideo", "-pix_fmt", "yuv420p",
                            yuv_dec,
                        ]
                        subprocess.run(decode_cmd, check=True,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                        # Quality (LPIPS)
                        y_dec, cb_dec, cr_dec = read_yuv420_clip_to_planes(
                            yuv_dec, T, H_enc, W_enc
                        )
                        dec_bgr = yuv420_planes_to_bgr_clip(y_dec, cb_dec, cr_dec)
                        V_cur = compute_quality_metric(ref_bgr, dec_bgr)

                        # Bitrate
                        video_size_kb = parse_ffmpeg_log_for_video_size(log_path)
                        duration_sec = T / float(fps)
                        if video_size_kb > 0:
                            bitrate_kbps = video_size_kb * 1024 * 8 / 1000.0 / duration_sec
                        else:
                            bitrate_kbps = os.path.getsize(bitstream) * 8.0 / duration_sec / 1000.0

                        R_dict[qp].append(bitrate_kbps)
                        V_dict[qp].append(V_cur)

        # Average over QPs
        results[vid] = {
            "R": [float(np.mean(R_variant_per_qp[qp])) for qp in qp_list],
            "V": [float(np.mean(V_variant_per_qp[qp])) for qp in qp_list],
        }
        baseline_results[vid] = {
            "R": [float(np.mean(R_baseline_per_qp[qp])) for qp in qp_list],
            "V": [float(np.mean(V_baseline_per_qp[qp])) for qp in qp_list],
        }

    return results, baseline_results


# ================================================================
# Main
# ================================================================
def main():
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    print("=" * 70)
    print(f"  Ablation Study Runner ({timestamp})")
    print(f"  {len(VARIANTS)} variants, {NUM_ITERS} iterations each")
    print("=" * 70)

    video_meta_list = VIDEO_META_LIST
    qp_list = CONFIG["qp_list"]

    # ============================================================
    # Phase 1: Train 3 variants sequentially
    # ============================================================
    variant_model_dirs = {}

    for vi, var in enumerate(VARIANTS):
        print("\n" + "=" * 70)
        print(f"  [{vi+1}/{len(VARIANTS)}] Training variant: {var['name']}")
        print(f"  Description: {var['desc']}")
        print(f"  w_mse={var['w_mse']}, w_hf={var['w_hf']}")
        print("=" * 70 + "\n")

        # Override CONFIG
        CONFIG["w_mse"] = var["w_mse"]
        CONFIG["w_hf"]  = var["w_hf"]
        CONFIG["num_iterations"] = NUM_ITERS
        CONFIG["quality_metric"] = QUALITY_METRIC
        CONFIG["log_dir"]   = os.path.join(OUTPUT_ROOT, f"runs/ablation_{timestamp}_{var['name']}")
        CONFIG["model_dir"] = os.path.join(OUTPUT_ROOT, f"checkpoints/ablation_{timestamp}_{var['name']}")

        variant_model_dirs[var["name"]] = CONFIG["model_dir"]

        t0 = time.time()
        train_rapnet_with_config(
            video_meta_list=video_meta_list,
            rd_json_path=RD_JSON_PATH,
        )
        elapsed = time.time() - t0
        print(f"\n  [OK] Variant {var['name']} training done, elapsed {elapsed/60:.1f} min\n")

    # ============================================================
    # Phase 2: Evaluate all variants and compute BD-LPIPS
    # ============================================================
    print("\n" + "=" * 70)
    print("  Phase 2: BD-LPIPS Evaluation")
    print("=" * 70)

    all_results = {}
    baseline_rd = None  # All variants share the same baseline (no Actor)

    for var in VARIANTS:
        name = var["name"]
        model_dir = variant_model_dirs[name]
        actor_path = os.path.join(model_dir, "actor_best.pth")

        if not os.path.exists(actor_path):
            print(f"  [WARN] {name}: actor_best.pth not found, skipping eval")
            continue

        print(f"\n  Evaluating variant: {name} ({actor_path})")
        results, baseline_results = evaluate_variant_rd(
            actor_path=actor_path,
            video_meta_list=video_meta_list,
            eval_video_ids=EVAL_VIDEO_IDS,
            qp_list=qp_list,
            clip_len=CONFIG["clip_len"],
            crop_h=CONFIG["crop_height"],
            crop_w=CONFIG["crop_width"],
            fps=CONFIG["fps"],
            ffmpeg_bin=CONFIG["ffmpeg_bin"],
            num_samples=5,
        )
        all_results[name] = results

        if baseline_rd is None:
            baseline_rd = baseline_results

    # ============================================================
    # Phase 3: Compute BD-LPIPS and print comparison table
    # ============================================================
    print("\n" + "=" * 70)
    print("  BD-LPIPS Comparison Results")
    print("=" * 70)

    # CSV output
    csv_path = os.path.join(OUTPUT_ROOT, f"ablation_results_{timestamp}.csv")
    csv_rows = []

    header = ["Video"] + [v["name"] + "_BD-Rate(%)" for v in VARIANTS] + \
             [v["name"] + "_BD-LPIPS" for v in VARIANTS]
    csv_rows.append(header)

    # Per-video computation
    bd_rate_all = {v["name"]: [] for v in VARIANTS}
    bd_lpips_all = {v["name"]: [] for v in VARIANTS}

    eval_vids = [vid for vid in EVAL_VIDEO_IDS if vid in (baseline_rd or {})]

    for vid in eval_vids:
        row = [vid]
        R_base = baseline_rd[vid]["R"]
        V_base = baseline_rd[vid]["V"]

        for var in VARIANTS:
            name = var["name"]
            if name in all_results and vid in all_results[name]:
                R_var = all_results[name][vid]["R"]
                V_var = all_results[name][vid]["V"]

                bdr = bd_rate(R_base, V_base, R_var, V_var)
                bdq = bd_quality(R_base, V_base, R_var, V_var)

                bd_rate_all[name].append(bdr)
                bd_lpips_all[name].append(bdq)

                row.append(f"{bdr:.2f}")
            else:
                row.append("N/A")
                bd_rate_all[name].append(0.0)

        for var in VARIANTS:
            name = var["name"]
            if name in all_results and vid in all_results[name]:
                R_var = all_results[name][vid]["R"]
                V_var = all_results[name][vid]["V"]
                bdq = bd_quality(R_base, V_base, R_var, V_var)
                row.append(f"{bdq:.4f}")
            else:
                row.append("N/A")
                bd_lpips_all[name].append(0.0)

        csv_rows.append(row)

    # Average row
    avg_row = ["Average"]
    for var in VARIANTS:
        name = var["name"]
        vals = bd_rate_all[name]
        avg_row.append(f"{np.mean(vals):.2f}" if vals else "N/A")
    for var in VARIANTS:
        name = var["name"]
        vals = bd_lpips_all[name]
        avg_row.append(f"{np.mean(vals):.4f}" if vals else "N/A")
    csv_rows.append(avg_row)

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in csv_rows:
            writer.writerow(row)
    print(f"\n  Results saved to: {csv_path}\n")

    # Print table
    print("\n" + "=" * 60)
    print("       Ablation Study BD-LPIPS Comparison")
    print("=" * 60)
    print(f"  {'Variant':<15} {'BD-Rate(%)':>12} {'BD-LPIPS':>12}")
    print("-" * 60)
    for var in VARIANTS:
        name = var["name"]
        bdr_avg = np.mean(bd_rate_all[name]) if bd_rate_all[name] else 0
        bdl_avg = np.mean(bd_lpips_all[name]) if bd_lpips_all[name] else 0
        print(f"  {name:<15} {bdr_avg:>+12.2f} {bdl_avg:>+12.4f}")
    print("=" * 60)

    # Save detailed results JSON
    detail_json = os.path.join(OUTPUT_ROOT, f"ablation_detail_{timestamp}.json")
    detail = {
        "timestamp": timestamp,
        "num_iters": NUM_ITERS,
        "variants": {v["name"]: {"w_mse": v["w_mse"], "w_hf": v["w_hf"]} for v in VARIANTS},
        "results": all_results,
        "baseline": baseline_rd,
        "bd_rate": {k: v for k, v in bd_rate_all.items()},
        "bd_lpips": {k: v for k, v in bd_lpips_all.items()},
    }
    with open(detail_json, "w", encoding="utf-8") as f:
        json.dump(detail, f, indent=2, ensure_ascii=False)
    print(f"  Detail JSON: {detail_json}")
    print(f"\n  [DONE] All ablation experiments completed!")


if __name__ == "__main__":
    main()
