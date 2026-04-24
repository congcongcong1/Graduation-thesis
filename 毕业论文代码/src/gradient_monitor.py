"""
梯度监控工具
用于调试Actor-Critic训练过程中的梯度流动问题
"""

import torch
import torch.nn as nn
from typing import Dict, List
import numpy as np


class GradientMonitor:
    """监控网络梯度的工具类"""

    def __init__(self, model: nn.Module, name: str = "Model"):
        self.model = model
        self.name = name
        self.grad_history = []

    def check_gradients(self) -> Dict[str, float]:
        """
        检查所有参数的梯度统计信息
        
        Returns:
            stats: 包含梯度统计的字典
              {
                'mean': 梯度平均值的绝对值,
                'std': 梯度标准差,
                'max': 最大梯度,
                'min': 最小梯度,
                'num_zero': 梯度为0的参数数量,
                'num_nan': 梯度为NaN的参数数量,
              }
        """
        grads = []
        num_zero = 0
        num_nan = 0

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach().cpu().numpy().flatten()

                # 统计零梯度
                if np.allclose(grad, 0.0):
                    num_zero += 1
                    print(f"[Warn] Zero gradient: {name}")

                # 统计NaN梯度
                if np.isnan(grad).any():
                    num_nan += 1
                    print(f"[Error] NaN gradient: {name}")

                grads.append(grad)
            else:
                print(f"[Warn] No gradient: {name}")

        if len(grads) == 0:
            print(f"[Error] {self.name}: No gradients found!")
            return {
                'mean': 0.0,
                'std': 0.0,
                'max': 0.0,
                'min': 0.0,
                'num_zero': 0,
                'num_nan': 0,
            }

        all_grads = np.concatenate(grads)

        stats = {
            'mean': float(np.mean(np.abs(all_grads))),
            'std': float(np.std(all_grads)),
            'max': float(np.max(all_grads)),
            'min': float(np.min(all_grads)),
            'num_zero': num_zero,
            'num_nan': num_nan,
        }

        return stats

    def print_stats(self):
        """打印梯度统计信息"""
        stats = self.check_gradients()

        print(f"\n{'='*60}")
        print(f"[{self.name}] Gradient Statistics:")
        print(f"{'='*60}")
        print(f"  Mean (abs): {stats['mean']:.6e}")
        print(f"  Std:        {stats['std']:.6e}")
        print(f"  Max:        {stats['max']:.6e}")
        print(f"  Min:        {stats['min']:.6e}")
        print(f"  Num Zero:   {stats['num_zero']}")
        print(f"  Num NaN:    {stats['num_nan']}")
        print(f"{'='*60}\n")

        # 记录历史
        self.grad_history.append(stats)

    def check_gradient_flow(self, threshold: float = 1e-7) -> bool:
        """
        检查梯度是否正常流动

        Args:
            threshold: 梯度均值的最小阈值

        Returns:
            True if gradients are flowing normally
        """
        stats = self.check_gradients()

        # 检查是否有梯度
        if stats['mean'] == 0.0:
            print(f"[Error] {self.name}: No gradient flow!")
            return False

        # 检查是否有NaN
        if stats['num_nan'] > 0:
            print(f"[Error] {self.name}: NaN gradients detected!")
            return False

        # 检查梯度是否过小
        if stats['mean'] < threshold:
            print(f"[Warn] {self.name}: Gradients very small (mean={stats['mean']:.2e})")
            return False

        # 检查梯度是否过大(可能爆炸)
        if stats['max'] > 1e3:
            print(f"[Warn] {self.name}: Gradients may be exploding (max={stats['max']:.2e})")
            return False

        print(f"[OK] {self.name}: Gradient flow is normal")
        return True


def register_hooks_for_gradient_monitoring(model: nn.Module, name: str = "Model"):
    """
    为模型注册hook以监控每层的梯度

    Args:
        model: 要监控的模型
        name: 模型名称
    """
    def make_hook(layer_name):
        def hook(grad):
            if torch.isnan(grad).any():
                print(f"[NaN Grad] {name}.{layer_name}")
            if torch.isinf(grad).any():
                print(f"[Inf Grad] {name}.{layer_name}")

            grad_norm = grad.norm().item()
            if grad_norm < 1e-7:
                print(f"[Vanishing Grad] {name}.{layer_name}: norm={grad_norm:.2e}")
            elif grad_norm > 1e3:
                print(f"[Exploding Grad] {name}.{layer_name}: norm={grad_norm:.2e}")

        return hook

    for layer_name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(make_hook(layer_name))


def diagnose_ac_training(actor: nn.Module,
                        critic: nn.Module,
                        critic_for_actor: nn.Module,
                        loss_actor: torch.Tensor,
                        loss_critic: torch.Tensor):
    """
    诊断Actor-Critic训练问题

    Args:
        actor: Actor网络
        critic: Critic网络
        critic_for_actor: Actor侧的Critic副本
        loss_actor: Actor损失
        loss_critic: Critic损失
    """
    print("\n" + "="*60)
    print("Actor-Critic Training Diagnosis")
    print("="*60)

    # 1. 检查损失值
    print("\n[1] Loss Values:")
    print(f"  Loss Actor:  {loss_actor.item():.6f}")
    print(f"  Loss Critic: {loss_critic.item():.6f}")

    if torch.isnan(loss_actor) or torch.isnan(loss_critic):
        print("  ❌ NaN loss detected!")
    else:
        print("  ✅ Loss values are normal")

    # 2. 检查Actor梯度
    print("\n[2] Actor Gradients:")
    actor_monitor = GradientMonitor(actor, "Actor")
    actor_ok = actor_monitor.check_gradient_flow()

    if not actor_ok:
        print("  ❌ Actor gradient flow issue detected!")
    else:
        print("  ✅ Actor gradients are flowing")

    # 3. 检查Critic梯度
    print("\n[3] Critic Gradients:")
    critic_monitor = GradientMonitor(critic, "Critic")
    critic_ok = critic_monitor.check_gradient_flow()

    if not critic_ok:
        print("  ❌ Critic gradient flow issue detected!")
    else:
        print("  ✅ Critic gradients are flowing")

    # 4. 检查critic_for_actor参数是否被冻结
    print("\n[4] Critic-for-Actor Parameters:")
    trainable_count = sum(p.requires_grad for p in critic_for_actor.parameters())
    total_count = sum(1 for _ in critic_for_actor.parameters())

    print(f"  Trainable params: {trainable_count}/{total_count}")

    if trainable_count > 0:
        print("  ❌ Critic-for-Actor should be frozen!")
    else:
        print("  ✅ Critic-for-Actor is correctly frozen")

    # 5. 检查参数同步
    print("\n[5] Parameter Synchronization:")
    params_synced = True
    for (n1, p1), (n2, p2) in zip(
        critic.named_parameters(),
        critic_for_actor.named_parameters()
    ):
        # ✅ 修复：将 p2 临时搬运到 p1 的设备上再比较
        if not torch.allclose(p1, p2.to(p1.device), atol=1e-6):
            print(f"  ❌ Params not synced: {n1}")
            params_synced = False

    if params_synced:
        print("  ✅ Critic and Critic-for-Actor are synced")
    else:
        print("  ❌ Parameter sync issue detected!")

    print("\n" + "="*60)
    print("Diagnosis Complete")
    print("="*60 + "\n")

def grad_stats(model: nn.Module) -> Dict[str, float]:
    """
    返回可写入 TensorBoard 的梯度统计
    """
    total = 0
    has_grad = 0
    nan_cnt = 0
    zero_cnt = 0
    l2_sq = 0.0
    max_abs = 0.0

    for _, p in model.named_parameters():
        total += 1
        if p.grad is None:
            continue
        has_grad += 1
        g = p.grad.detach()
        if torch.isnan(g).any():
            nan_cnt += 1
        if torch.all(g == 0):
            zero_cnt += 1
        l2_sq += float(torch.sum(g * g).item())
        max_abs = max(max_abs, float(torch.max(torch.abs(g)).item()))

    l2 = float(l2_sq ** 0.5)
    return {
        "grad_l2": l2,
        "grad_max_abs": max_abs,
        "grad_has_grad_ratio": (has_grad / total) if total > 0 else 0.0,
        "grad_nan_param_cnt": float(nan_cnt),
        "grad_zero_param_cnt": float(zero_cnt),
    }
    
if __name__ == "__main__":
    # 示例用法
    print("Gradient Monitoring Utilities")
    print("\n使用方法:")
    print("""
    from gradient_monitor import GradientMonitor, diagnose_ac_training

    # 在训练循环中
    loss_actor.backward()
    loss_critic.backward()

    # 诊断梯度问题
    diagnose_ac_training(
        actor=agent.actor,
        critic=agent.critic,
        critic_for_actor=agent.critic_for_actor,
        loss_actor=loss_actor,
        loss_critic=loss_critic
    )
    """)
