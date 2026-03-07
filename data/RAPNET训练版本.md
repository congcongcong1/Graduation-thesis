版本说明：
# ElegantRL-master/my_RL/runs/rapnet_hevc v1版本：
损失为pixel_loss = 6.0 * pixel_loss_y + 1*pixel_loss_cb + 1*pixel_loss_cr
出现问题：cb通道直接全部被切断为0

# ElegantRL-master/my_RL/runs/rapnet_hevc_12_12_V2 v2版本：
损失为pixel_loss = 6.0 * pixel_loss_y + 3*pixel_loss_cb + 3*pixel_loss_cr
出现问题：cb通道直![alt text](image.png)接全部被切断为0,并且码率出现暴涨的情况

# ElegantRL-master/my_RL/runs/rapnet_hevc_12_13_v3 v3版本：
新增函数 total_variation_loss 这个 Loss 会迫使 Actor 生成“干净”的图像，直接解决码率暴涨的问题
修改 RAPNet 输出层：使用 Tanh 限制 Delta 幅度 确保不会让他做出切断某个通道的操作
出现问题：actor300步就开始停滞学习，直接采用原图，发现是因为baseline采用的是全图取平均的方式，但是这很不公平，
训练时我们只切取 16 帧进行编码，这会导致第一个帧必须是 I 帧。对于短 Clip，I 帧的巨大比特数会被平摊到很少的帧上，导致平均码率远高于长视频。如果用“整条视频”做 Baseline，训练时的码率永远会比 Baseline 高，导致 Reward 恒为负

# ElegantRL-master/my_RL/runs/rapnet_hevc_12_13_v4 v4版本：
新增基于“随机 Clip 采样”的 Baseline RD 构建函数，修复不公平现象

# ElegantRL-master/my_RL/runs/rapnet_hevc_12_15_v5 v5版本：
暂时移除 tv_loss,改变超参数：alpha_q = 0.4  beta_r  = 1.0  reward *= 10这会让“省码率”有机会抵消一点质量劣势，同时放大奖励,同时将lamda_pixel改为0.01
新增探索噪声，随着训练进行，噪声可以衰减。这里先给一个固定的 Ysigma=0.05 CbCr=0.02(对应像素值范围 0-1)

# ElegantRL-master/my_RL/runs/rapnet_hevc_12_15_v6 v6版本：
对超参数进行了微调，lamada_pixel=0.01,探索噪声强度noise_y_scale = 0.005  # 从 0.05 降到 0.01
delta = torch.tanh(delta) * 0.05
noise_c_scale = 0.002 # 从 0.02 降到 0.005，同时用tanh截断 Reward，防止极端负值,
alpha_q = 1.0   # 100% 关注质量
beta_r  = 0.05  # 5% 关注码率 (几乎忽略码率增加的惩罚)

# ElegantRL-master/my_RL/runs/rapnet_hevc_12_16_v7 v7版本：
1. 放宽 delta 的限制，原来是 * 0.05 (太小了)，现在改为 * 0.5，
2. 同时移除 forward 内部的硬截断 (torch.clamp)，允许暂时越界，在act函数中目的：因为网络现在输出的是 Raw 值（可能越界），在送给环境（FFmpeg）之前，必须手动截断到 [0, 1]，否则编码器会报错或产生奇怪颜色。
3. 修改 Agent 的 Update 函数 (核心：加入饱和惩罚)位置：class DeterministicACAgentYCbCr -> update 函数 目的：在这里处理 Raw 输出。用 Hinge Loss 惩罚越界行为，同时把 Raw 值 Clamp 后再喂给 Critic（因为 Critic 需要看“真实的画面”）。


# ElegantRL-master/my_RL/runs/rapnet_hevc_12_16_v8 v8版本: 函数：my_RL/rapnet_hevc_ppo_qp_yuv_simple.py
改回最初始的最简单的状态，去除噪声，只保留baseline_clip的更改，alpha_q = 1.0   # 100% 关注质量
beta_r  = 0.05  # 5% 关注码率  同时去掉tv_loss等loss约束，只有pixel_loss约束，约束值为0.01

# v8版本更改版：函数：my_RL/rapnet_hevc_ppo_qp_yuv_simple.py
更改：$\alpha_q = 4.0$ (画质权重)$\beta_r = 0.5$ (码率权重) delta = torch.tanh(delta) * 0.2截断
lambda_pixel=1.0

再次更改:专注于低码率场景：qp 37，40，42，45，47 rd_baseline_hevc_qp_full_clip_lowbits.json
lambda_pixel=0.1，降低对actor的束缚，由于critic梯度爆炸，于是对其进行了梯度裁剪限制

# v9版本在前面的基础上，改为delta = torch.tanh(delta)*0.02 0.2 (50灰度级) 依然能破坏画质。我们把它限制在 0.02 (约 5 个灰度级)。 让他学习如何去除纹理

# v10版本在前面的基础上，  【核心修改】零均值约束对 B, C, H, W 的每个样本、每个通道，减去该通道的空间均值 delta = delta - delta.mean(dim=(2, 3), keepdim=True)，然后将belta的权重调到1
# v11版本在前面的基础上，为了修复棋盘格，加入tv_loss约束

# v12版本在前面的基础上为了解决某一部暴跌的问题，将学习率改为了1e-5，同时去掉两个高码率视频readysteadygo_1080p     yachtride_1080p 同时对YCbCr三通道都加权重约束

# v13版本：
1. 降低 lambda_pixel = 0.01 (从 0.1 下调)。原因：数学诊断显示 v12 中像素损失的“收缩力”是 Critic 建议梯度的 60-100 倍，导致 Actor 被锁死在极小改动量（0.0035），无法有效探索。下调后两者梯度处于同一量级。
2. 启用 delta = torch.tanh(delta) * 0.1。原因：在放开 lambda_pixel 约束后，使用 Tanh 作为“安全带”物理限制改动幅度，同时保证梯度在小范围内更连续，避免训练跑飞。
3. 协同进化：建议从 0 开始重新训练，让 Actor 和 Critic 在新的物理规则下重新建立 RD 平衡。

# v14版本
取消 delta = torch.tanh(delta) * 0.1
根据服务器实际，也更改了GPU的设定，下次训练要记得改回来

# 15版本
延续v13版本的最佳权重继续训练
更改：取消 delta = torch.tanh(delta) * 0.1    调整alpha_q =2
ps：v14与v15是两个同步的尝试！

# 16版本
v15版本暴跌
更改：恢复delta = torch.tanh(delta) * 0.1 调整alpha_q=2  lambda_pixel =0.001

test_rapnet.py用于加载模型来验证

# v17版本
# 核心修改：将奖励权重提高
    "alpha_q": 5.0,    # 从 2.0 提高到 5.0 (2.5倍)，增强画质保护强度
    "beta_r": 3.0,     # 从 1.0 提高到 3.0 (3倍)，更积极地探索码率节省
    # 这允许 Actor 改变像素的最大幅度从 10% 提升到 20%，跨过编码器的敏感度阈值
    delta = torch.tanh(delta) * 0.2
    lambda_pixel =0.0005

# v18 采用非对称奖励的方式
1  非对称奖励 如果画质差，权重变小；如果画质好，权重保持
2  直接去除tvloss
3  act中训练时强行加噪声0.02 约5个灰度等级
4  "alpha_q": 5.0,    # 从 2.0 提高到 5.0 (2.5倍)，增强画质保护强度
   "beta_r": 5.0,     # 从 1.0 提高到 5.0 (5倍)，更积极地探索码率节省

# v19  在v18的基础上加入tv_loss,让模型学会去除高频信息

# v20 彻底抛弃pixel_loss 改为使用 Loss (Downsampled MSE) 来束缚低频
    "alpha_q": 3.0,
    "beta_r": 5.0,
    同时调整 Hinge 非对称奖励，对奖励加截断reward_qp = torch.clamp(reward_qp_raw, -5.0, 5.0)
    去除 act中训练时强行加的噪声，全部相信tv_loss的平滑能力
    loss_actor = -q_for_actor.mean() + 0.01 * pixel_loss_low_freq + 0.05*tv_loss_content

# v21 由于v20还是倾向于学一些低频的信息
        loss_actor = -q_for_actor.mean() + \
                     0.05 * pixel_loss_simple + \
                     0.02 * tv_loss_simple
从 v20的 300 step 恢复


# v22  v21从step800开始崩溃
修改：reward不再是断崖式惩罚，而是平滑线惩罚
修改loss还是为MSE束缚，看能否有效
    "alpha_q": 10,
    "beta_r": 2,  码率允许提升， 但是画质更重要
    reward_base = 10.0 * delta_v + 2.0 * delta_r

# v23  v22依旧高频修改为0
目标：增强低频结构，削减中高频纹理
加入img中的diff频带分析，确保和论文中一致
    "alpha_q": 5.0,
    "beta_r": 5.0,

# v24
# 1. 回归质量优先 (8:2)
 reward_base = 8.0 * delta_v + 2.0 * delta_r
# 2. 新增边缘检测损失函数
loss_actor = -q_for_actor.mean() + \
                     0.1 * loss_pixel + \
                     0.02 * tv_loss_simple + \
                     0.01 * loss_edge(使用各向异性的sable算子，而不是laplace)
# 3. 简化惩罚逻辑
penalty = 0.0
if delta_v < -0.02:  # 门槛从 5% 缩减到 2%
penalty = 20.0 * abs(delta_v)  # 使用线性惩罚，更平滑

# v25  仿造论文实现，增强低频结构，削减中高频纹理
        loss_actor = -q_for_actor.mean() \
                    + w_low  * loss_low \
                    + w_hf   * loss_hf \
                    + w_edge * loss_edge_focus \
                    + w_tv   * tv_loss \
                    + w_id   * loss_id
        reward依旧是画质要求优先于码率

# v26 去掉l1loss 增强低频结构
改为码率优先于画质，因为画质实在是不好改动
loss_mse系数为2000
reward_final = 100 * delta_r + 50 * delta_v  # 基础奖励：码率节省部分
loss_actor = -q_for_actor.mean() + w_mse * loss_mse

# v27  loss_mse系数降低为2

# v28
 reward_final = 200.0 * delta_r + 20.0 * delta_v
 w_mse = 0.3

# v29
delta = torch.tanh(delta) * 0.05
w_mse = 5.0
reward_final = 200.0 * delta_r + 50.0 * delta_v

# v30
去掉均值约束,使他能修改低频
# v31
调整了奖励函数，用最初的最简单的 reward_final = 200.0 * delta_r + 50.0 * delta_v

# v32
reward_final = 50 * delta_r + 200 * delta_v

# v34 35
模型初始化的时候限制weight bias=0 （仅针对actor网络） 以应对残差连接

# v37
w_mse = 5 w_hf = 10
delta = torch.tanh(delta) * 0.05

# v38  均是在试系数
        w_lpips = 5.0        # 主力（替代 v37 的 MSE）
        w_hf = 10.0          # 辅助（与 v37 相同）
        w_diagonal = 15.0    # 辅助（新增，抑制棋盘格）
        引入loss_lpips,以及loss_diagonal_focused对抗四角高频

# v39
        w_lpips = 0.005        # 降低（5 → 0.005）
        w_hf = 0.0           # 移除（允许全局增强）
        w_diagonal = 5.0     # 降低（15 → 5，因为移除了 mask）

# v43 v44     不再上下采样，而是采用random——crop到540p 代码文件：1080p结尾
w_mse =0.5 w_hf =1.0

# v45
w_mse =0.1 w_hf =0.5

# 2_9_v46
w_mse=0.3  w_hf =0.5
"lpips","vmaf", "dists", "wasserstein"  分别进行1k次训练

# 2_11_v47   -代码文件layer_clip为引入分层以及clip大模型评估代码
w_mse =5 w_hf =10
