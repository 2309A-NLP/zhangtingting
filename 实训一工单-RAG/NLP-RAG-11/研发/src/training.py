"""
训练参数与训练流程模块

定义训练参数配置、训练循环，支持多种损失函数和数据集格式。
"""

import os
import json
import time
import torch
from typing import Dict, List, Optional, Any, Tuple
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.evaluation import (
    SentenceEvaluator,
    TripletEvaluator,
    EmbeddingSimilarityEvaluator,
    InformationRetrievalEvaluator,
)


class TrainingConfig:
    """训练配置类"""
    
    def __init__(
        self,
        # 训练参数
        num_epochs: int = 3,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        
        # 优化器与调度器
        optimizer_class: str = "AdamW",
        scheduler: str = "warmupcosine",
        
        # 保存与日志
        output_dir: str = "./models/finetuned",
        save_best_model: bool = True,
        evaluation_steps: int = 100,
        
        # 其他
        max_seq_length: Optional[int] = None,
        use_amp: bool = True,  # 混合精度训练
        
        # Matryoshka 相关
        use_matryoshka: bool = False,
        matryoshka_sizes: Optional[List[int]] = None,
        matryoshka_weight: float = 0.5,
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.optimizer_class = optimizer_class
        self.scheduler = scheduler
        self.output_dir = output_dir
        self.save_best_model = save_best_model
        self.evaluation_steps = evaluation_steps
        self.max_seq_length = max_seq_length
        self.use_amp = use_amp
        self.use_matryoshka = use_matryoshka
        self.matryoshka_sizes = matryoshka_sizes
        self.matryoshka_weight = matryoshka_weight
    
    def __repr__(self) -> str:
        params = [f"{k}={v}" for k, v in self.__dict__.items()]
        return f"TrainingConfig({', '.join(params)})"


def create_trainer(
    model: SentenceTransformer,
    train_dataloader: DataLoader,
    loss_function: torch.nn.Module,
    evaluator: Optional[SentenceEvaluator] = None,
    config: Optional[TrainingConfig] = None,
    **kwargs,
):
    """
    创建 SentenceTransformer 训练器并执行训练
    
    使用 SentenceTransformer 的 fit() 方法进行训练。
    """
    if config is None:
        config = TrainingConfig()
    
    # 设置最大序列长度
    if config.max_seq_length:
        model.max_seq_length = config.max_seq_length
    
    # 准备训练参数
    train_args = {
        "train_objectives": [(train_dataloader, loss_function)],
        "epochs": config.num_epochs,
        "warmup_steps": config.warmup_steps,
        "optimizer_params": {
            "lr": config.learning_rate,
            "weight_decay": config.weight_decay,
        },
        "output_path": config.output_dir,
        "save_best_model": config.save_best_model,
        "use_amp": config.use_amp,
        "show_progress_bar": True,
    }
    
    if evaluator is not None:
        train_args["evaluator"] = evaluator
        train_args["evaluation_steps"] = config.evaluation_steps
    
    # 如果指定了调度器
    if config.scheduler:
        train_args["scheduler"] = config.scheduler
    
    # 额外参数
    train_args.update(kwargs)
    
    print("=" * 60)
    print("Starting Training")
    print("=" * 60)
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Warmup steps: {config.warmup_steps}")
    print(f"  Weight decay: {config.weight_decay}")
    print(f"  Optimizer: {config.optimizer_class}")
    print(f"  Scheduler: {config.scheduler}")
    print(f"  Output dir: {config.output_dir}")
    print(f"  Max seq length: {model.max_seq_length}")
    print(f"  Use AMP: {config.use_amp}")
    print(f"  Use Matryoshka: {config.use_matryoshka}")
    print("=" * 60)
    
    start_time = time.time()
    model.fit(**train_args)
    elapsed = time.time() - start_time
    
    print(f"\nTraining completed in {elapsed:.2f}s ({elapsed/60:.2f}min)")
    print(f"Model saved to: {config.output_dir}")
    
    return model


def save_training_config(config: TrainingConfig, filepath: str):
    """保存训练配置到 JSON 文件"""
    config_dict = {
        "num_epochs": config.num_epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "warmup_steps": config.warmup_steps,
        "weight_decay": config.weight_decay,
        "optimizer": config.optimizer_class,
        "scheduler": config.scheduler,
        "output_dir": config.output_dir,
        "save_best_model": config.save_best_model,
        "evaluation_steps": config.evaluation_steps,
        "max_seq_length": config.max_seq_length,
        "use_amp": config.use_amp,
        "use_matryoshka": config.use_matryoshka,
        "matryoshka_sizes": config.matryoshka_sizes,
        "matryoshka_weight": config.matryoshka_weight,
    }
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)
    print(f"Training config saved to {filepath}")


def load_training_config(filepath: str) -> TrainingConfig:
    """从 JSON 文件加载训练配置"""
    with open(filepath, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    return TrainingConfig(**config_dict)


if __name__ == "__main__":
    # 测试
    config = TrainingConfig()
    print("Default TrainingConfig:")
    print(f"  num_epochs={config.num_epochs}, batch_size={config.batch_size}")
    print(f"  learning_rate={config.learning_rate}, warmup_steps={config.warmup_steps}")
