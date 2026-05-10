# 毕业论文代码说明

本目录为论文《基于强化学习的视频编码前处理方案》的代码附件。源码、评估脚本与模型权重已按提交用途整理，论文附录中的“核心代码文件目录说明”和“代码提交说明”与本目录保持一致。

## 目录结构

```text
毕业论文代码/
|-- src/
|   |-- rapnet_hevc_ppo_qp_yuv_1080p.py    # 最终1080p Random Crop训练脚本
|   |-- rapnet_hevc_ppo_qp_yuv_simple.py   # 下采样版本训练/评估脚本
|   |-- gradient_monitor.py                # Actor-Critic梯度诊断工具
|-- tools/
|   |-- eval_rapnet_full_video.py          # 完整视频推理与RD评估
|   |-- evaluate_time.py                   # 端到端耗时统计
|   |-- run_ablation_3variants.py          # 三组损失配置消融实验
|-- checkpoints/
|   |-- actor_best.pth                     # Actor最优权重
|   |-- critic_best.pth                    # Critic最优权重
|-- requirements.txt                       # Python依赖
```

## 环境要求

- Python 3.10+
- PyTorch 2.0+
- NVIDIA GPU与CUDA环境
- FFmpeg 6.0+，需支持`hevc_nvenc`

安装Python依赖：

```bash
pip install -r requirements.txt
```

## 运行方式

训练最终Random Crop版本：

```bash
python src/rapnet_hevc_ppo_qp_yuv_1080p.py
```

完整视频评估：

```bash
python tools/eval_rapnet_full_video.py
```

端到端耗时统计：

```bash
python tools/evaluate_time.py
```

消融实验：

```bash
python tools/run_ablation_3variants.py
```

## 路径配置

标准测试序列体积较大，未随代码附件提交。复现实验前需要将`VIDEO_META_LIST`中的YUV路径改为本机数据集路径，或保持脚本默认的`/datasets/...`目录结构。

常用环境变量如下：

- `RAPNET_ACTOR_CKPT`：Actor权重路径，默认使用`checkpoints/actor_best.pth`
- `RAPNET_BASELINE_JSON`：Baseline RD曲线JSON路径，默认从`results/`读取
- `RAPNET_DEVICE`：运行设备，例如`cuda:0`或`cpu`
- `RAPNET_RESULT_DIR`：训练脚本结果目录
- `RAPNET_RUN_DIR`：TensorBoard日志目录
- `RAPNET_CHECKPOINT_DIR`：训练输出权重目录
- `RAPNET_TMP_DIR`：FFmpeg临时文件目录
- `RAPNET_VIDEO_PATH`：`tools/evaluate_time.py`使用的单视频路径
