"""
NLP-RAG-11 Embeddings 模型微调 - 主运行脚本

流程:
1. 生成数据集
2. 加载模型 (BAAI/bge-base-en-v1.5)
3. 加载数据集
4. 定义损失函数
5. 配置训练参数
6. 微调前评估
7. 执行微调
8. 微调后评估
9. 对比结果
"""

import os
import sys
import json
import torch
import argparse
from datetime import datetime

# 确保能找到 src 模块
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)


def main():
    parser = argparse.ArgumentParser(description="NLP-RAG-11 Embedding Model Fine-tuning")
    parser.add_argument("--model_name", type=str, default="BAAI/bge-base-en-v1.5",
                        help="Base embedding model")
    parser.add_argument("--domain", type=str, default="legal",
                        choices=["legal", "medical", "psbc"],
                        help="Domain for fine-tuning")
    parser.add_argument("--loss_type", type=str, default="cosine_similarity",
                        choices=["cosine_similarity", "contrastive", "triplet",
                                 "batch_hard_triplet", "mnrl", "mnrsl", "cosent"],
                        help="Loss function type")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--use_matryoshka", action="store_true",
                        help="Use Matryoshka Loss")
    parser.add_argument("--no_eval_before", action="store_true",
                        help="Skip pre-fine-tuning evaluation")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Data directory (generated if not exists)")
    parser.add_argument("--data_prefix", type=str, default=None,
                        help="File prefix for training data (default: same as domain)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for models and results")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # ==================== 路径设置 ====================
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = args.data_dir or os.path.join(project_root, "data", "processed")
    output_dir = args.output_dir or os.path.join(project_root, "models", f"finetuned_{args.domain}")
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "=" * 70)
    print("   NLP-RAG-11 | Embeddings Model Fine-tuning")
    print("   " + "=" * 70)
    print(f"   Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Domain: {args.domain}")
    print(f"   Model: {args.model_name}")
    print(f"   Loss: {args.loss_type}")
    print("   " + "=" * 70)
    
    # ==================== 步骤 1: 生成数据集 ====================
    print("\n" + "=" * 60)
    print("  Step 1/9: Generating Domain-Specific Dataset")
    print("=" * 60)
    
    from src.data_generation import generate_all_datasets
    
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        generate_all_datasets(data_dir)
    else:
        print(f"Dataset already exists at {data_dir}, skipping generation.")
    
    # ==================== 步骤 2: 加载基础模型 ====================
    print("\n" + "=" * 60)
    print("  Step 2/9: Loading Base Model")
    print("=" * 60)
    
    from src.model_loading import load_model, load_all_training_data
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_name, device=device)
    
    # ==================== 步骤 3: 加载训练数据 ====================
    print("\n" + "=" * 60)
    print("  Step 3/9: Loading Training Data")
    print("=" * 60)
    
    data_loaders = load_all_training_data(data_dir, args.domain, args.batch_size,
                                           data_prefix=args.data_prefix)
    
    if not data_loaders:
        print("ERROR: No training data loaded. Aborting.")
        sys.exit(1)
    
    # ==================== 步骤 4: 定义损失函数 ====================
    print("\n" + "=" * 60)
    print("  Step 4/9: Defining Loss Function")
    print("=" * 60)
    
    from src.loss_functions import get_loss_function
    
    loss_fn = get_loss_function(
        model=model,
        loss_type=args.loss_type,
        device=device,
        use_matryoshka=args.use_matryoshka,
    )
    
    # ==================== 步骤 5: 配置训练参数 ====================
    print("\n" + "=" * 60)
    print("  Step 5/9: Configuring Training Parameters")
    print("=" * 60)
    
    from src.training import TrainingConfig, save_training_config
    
    config = TrainingConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=output_dir,
        use_matryoshka=args.use_matryoshka,
    )
    
    config_file = os.path.join(results_dir, f"training_config_{timestamp}.json")
    save_training_config(config, config_file)
    
    # ==================== 步骤 6: 微调前评估 ====================
    if not args.no_eval_before:
        print("\n" + "=" * 60)
        print("  Step 6/9: Evaluating Model BEFORE Fine-tuning")
        print("=" * 60)
        
        from src.evaluation import evaluate_model
        
        metrics_before = evaluate_model(
            model, domain=args.domain, name="Before Fine-tuning"
        )
    else:
        metrics_before = {}
        print("\nSkipping pre-fine-tuning evaluation.")
    
    # ==================== 步骤 7: 执行微调 ====================
    print("\n" + "=" * 60)
    print("  Step 7/9: Executing Fine-tuning")
    print("=" * 60)
    
    from src.training import create_trainer
    
    # 保存原始模型引用，用于 Step 9 对比
    original_model = model
    
    # 使用正例对数据进行训练
    train_loader = data_loaders.get("positive_pairs")
    if train_loader is None:
        # 如果没有正例对，使用三元组数据
        train_loader = list(data_loaders.values())[0]
    
    model = create_trainer(
        model=model,
        train_dataloader=train_loader,
        loss_function=loss_fn,
        config=config,
    )
    
    # ==================== 步骤 8: 微调后评估 ====================
    print("\n" + "=" * 60)
    print("  Step 8/9: Evaluating Model AFTER Fine-tuning")
    print("=" * 60)
    
    from src.evaluation import evaluate_model
    
    # 重新加载微调后的模型以确保评估正确
    from src.model_loading import load_model
    model_after = load_model(output_dir, device=device)
    
    metrics_after = evaluate_model(
        model_after, domain=args.domain, name="After Fine-tuning"
    )
    
    # ==================== 步骤 9: 对比结果 ====================
    print("\n" + "=" * 60)
    print("  Step 9/9: Comparing Results")
    print("=" * 60)
    
    from src.evaluation import compare_models
    
    comparison_file = os.path.join(results_dir, f"comparison_{timestamp}.json")
    comparison = compare_models(
        original_model,
        model_after,
        domain=args.domain,
        save_path=comparison_file,
    )
    
    # ==================== 生成图表 ====================
    try:
        import importlib.util
        chart_spec = importlib.util.spec_from_file_location(
            "gen_charts",
            os.path.join(os.path.dirname(__file__), "..", "_gen_charts.py")
        )
        if chart_spec and chart_spec.loader:
            chart_mod = importlib.util.module_from_spec(chart_spec)
            chart_spec.loader.exec_module(chart_mod)
            print("Charts generated automatically.")
    except Exception as e:
        print(f"Chart generation skipped ({e}).")

    # ==================== 完成 ====================
    print("\n" + "=" * 70)
    print("   Fine-tuning Complete!")
    print("   " + "=" * 70)
    print(f"   End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Model saved to: {output_dir}")
    print(f"   Results saved to: {comparison_file}")
    print(f"   Training config: {config_file}")
    print("   " + "=" * 70)
    
    # 生成最终验收概要
    summary_path = os.path.join(results_dir, f"summary_{timestamp}.json")
    summary = {
        "project": "NLP-RAG-11",
        "task": "Embeddings Model Fine-tuning",
        "domain": args.domain,
        "base_model": args.model_name,
        "loss_function": args.loss_type,
        "use_matryoshka": args.use_matryoshka,
        "training_config": {
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        },
        "evaluation_before": {
            k: float(v) if not isinstance(v, float) else v
            for k, v in metrics_before.items()
        },
        "evaluation_after": {
            k: float(v) if not isinstance(v, float) else v
            for k, v in metrics_after.items()
        },
        "improvement": {
            k: {
                "before": float(v["before"]),
                "after": float(v["after"]),
                "change_pct": float(v["percentage_change"]),
            }
            for k, v in comparison.get("improvement", {}).items()
        },
        "verification": "微调后模型检索效果优于微调前" if all(
            v["after"] >= v["before"]
            for v in comparison.get("improvement", {}).values()
        ) else "需进一步分析",
        "completion_time": datetime.now().isoformat(),
    }
    
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Summary saved to: {summary_path}")
    
    return summary


if __name__ == "__main__":
    main()
