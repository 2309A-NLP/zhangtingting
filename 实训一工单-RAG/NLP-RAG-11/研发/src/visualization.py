"""
结果可视化模块

生成微调前后对比图表、评估指标可视化。
"""

import json
import os
import sys
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import numpy as np


def plot_metrics_comparison(
    before_metrics: dict,
    after_metrics: dict,
    save_path: str,
    title: str = "Model Performance: Before vs After Fine-tuning",
):
    """绘制微调前后指标对比柱状图"""
    metrics = sorted(before_metrics.keys())
    
    before_values = [before_metrics[m] for m in metrics]
    after_values = [after_metrics[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, before_values, width, label='Before Fine-tuning', color='#6BAED6')
    bars2 = ax.bar(x + width/2, after_values, width, label='After Fine-tuning', color='#FD8D3C')
    
    ax.set_ylabel('Score')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    # 在柱子上标注数值
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Chart saved to {save_path}")
    plt.close()


def plot_improvement_chart(
    improvement_data: dict,
    save_path: str,
):
    """绘制改进幅度图"""
    metrics = sorted(improvement_data.keys())
    improvements = [improvement_data[m]["percentage_change"] for m in metrics]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#31A354' if v >= 0 else '#E6550D' for v in improvements]
    bars = ax.barh(metrics, improvements, color=colors)
    
    ax.set_xlabel('Improvement (%)')
    ax.set_title('Performance Improvement After Fine-tuning', fontsize=13, fontweight='bold')
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
    
    for bar, val in zip(bars, improvements):
        label = f'{val:+.2f}%'
        ax.annotate(label,
                    xy=(bar.get_width() + (0.5 if val >= 0 else -2.5), bar.get_y() + bar.get_height()/2),
                    ha='left' if val >= 0 else 'right',
                    va='center',
                    fontsize=9,
                    fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Improvement chart saved to {save_path}")
    plt.close()


def plot_training_loss(
    loss_history: list,
    save_path: str,
):
    """绘制训练损失曲线"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(loss_history, color='#3182BD', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Curve', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Loss curve saved to {save_path}")
    plt.close()


def generate_report_from_comparison(
    comparison_file: str,
    output_dir: str,
):
    """从对比结果 JSON 生成可视化报告"""
    with open(comparison_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    before = data.get("before", {})
    after = data.get("after", {})
    improvement = data.get("improvement", {})
    
    # 指标对比图
    plot_metrics_comparison(
        before, after,
        os.path.join(output_dir, "metrics_comparison.png")
    )
    
    # 改进幅度图
    plot_improvement_chart(
        improvement,
        os.path.join(output_dir, "improvement_chart.png")
    )
    
    print(f"\nVisual report generated in {output_dir}")
    print(f"  - metrics_comparison.png")
    print(f"  - improvement_chart.png")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        generate_report_from_comparison(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "./results")
    else:
        print("Usage: python visualization.py <comparison.json> [output_dir]")
