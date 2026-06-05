"""
模型与数据集加载模块

负责加载 BAAI/bge-base-en-v1.5 模型以及各格式的训练数据集。
"""

import json
import torch
import os
from typing import List, Tuple, Dict, Optional, Union
from sentence_transformers import SentenceTransformer, InputExample
from torch.utils.data import DataLoader, Dataset


# ==================== 模型加载 ====================

def _find_snapshot_path(model_name: str) -> Optional[str]:
    """
    Hub 缓存中查找最近的快照路径（因为 robocopy 破坏了 symlink 结构）。
    """
    hub_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    model_cache = os.path.join(hub_dir, f"models--{model_name.replace('/', '--')}")
    snapshots_dir = os.path.join(model_cache, "snapshots")
    if os.path.isdir(snapshots_dir):
        snaps = sorted(os.listdir(snapshots_dir))
        if snaps:
            snap_path = os.path.join(snapshots_dir, snaps[-1])
            # 检查是否存在 model.safetensors 或 pytorch_model.bin
            for fname in ["model.safetensors", "pytorch_model.bin", "config.json"]:
                if os.path.isfile(os.path.join(snap_path, fname)):
                    return snap_path
    return None


def load_model(
    model_name: str = "BAAI/bge-base-en-v1.5",
    device: Optional[str] = None,
    cache_folder: Optional[str] = None,
) -> SentenceTransformer:
    """
    加载 SentenceTransformer 模型
    
    Args:
        model_name: HuggingFace 模型名称
        device: 设备 ('cuda', 'cpu', 或 None 自动选择)
        cache_folder: 模型缓存目录
    
    Returns:
        SentenceTransformer 模型实例
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model: {model_name}")
    print(f"Device: {device}")
    
    # 先尝试本地快照路径（绕过 hub 缓存 symlink 问题）
    local_path = _find_snapshot_path(model_name)
    if local_path:
        print(f"Found local snapshot at: {local_path}")
        kwargs = {}
        if cache_folder:
            kwargs["cache_folder"] = cache_folder
        model = SentenceTransformer(local_path, **kwargs)
    else:
        kwargs = {}
        if cache_folder:
            kwargs["cache_folder"] = cache_folder
        model = SentenceTransformer(model_name, **kwargs)
    
    model = model.to(device)
    
    print(f"Model loaded. Max sequence length: {model.max_seq_length}")
    dim = getattr(model, "get_sentence_embedding_dimension", None) or model.get_embedding_dimension
    print(f"Embedding dimension: {dim()}")
    
    return model


# ==================== 数据集加载 ====================

def load_positive_pairs(
    filepath: str,
) -> List[InputExample]:
    """
    加载正例对数据集 (sentence1, sentence2, score)
    
    转换为 InputExample 格式用于 SentenceTransformer 训练。
    label 存储为 float 相似度分数。
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    examples = []
    for item in data:
        # item = (sentence1, sentence2, score)
        if isinstance(item, list) and len(item) >= 3:
            examples.append(InputExample(
                texts=[str(item[0]), str(item[1])],
                label=float(item[2])
            ))
    
    print(f"Loaded {len(examples)} positive pair examples from {filepath}")
    return examples


def load_triplets(
    filepath: str,
) -> List[InputExample]:
    """
    加载三元组数据集 (anchor, positive, negative)
    
    转换为 InputExample 格式，标签使用 triplet loss 格式。
    label=0 表示 anchor-positive 对 (相似)，
    label=1 表示 anchor-negative 对 (不相似)。
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    examples = []
    for item in data:
        if isinstance(item, list) and len(item) >= 3:
            anchor, positive, negative = str(item[0]), str(item[1]), str(item[2])
            # 正例对
            examples.append(InputExample(texts=[anchor, positive], label=0.0))
            # 负例对
            examples.append(InputExample(texts=[anchor, negative], label=1.0))
    
    print(f"Loaded {len(examples)} triplet-derived examples from {filepath}")
    return examples


def load_cosine_pairs(
    filepath: str,
) -> List[InputExample]:
    """
    加载带相似度分数的句子对数据集
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    examples = []
    for item in data:
        if isinstance(item, list) and len(item) >= 3:
            examples.append(InputExample(
                texts=[str(item[0]), str(item[1])],
                label=float(item[2])
            ))
    
    print(f"Loaded {len(examples)} cosine similarity examples from {filepath}")
    return examples


def create_dataloader(
    examples: List[InputExample],
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    """
    创建 PyTorch DataLoader
    """
    dataloader = DataLoader(examples, batch_size=batch_size, shuffle=shuffle)
    print(f"DataLoader created: {len(dataloader)} batches (batch_size={batch_size})")
    return dataloader


def load_all_training_data(
    data_dir: str,
    domain: str = "legal",
    batch_size: int = 32,
    data_prefix: str = None,
) -> Dict[str, DataLoader]:
    """
    加载指定领域的所有训练数据
    
    返回各类数据集的 DataLoader 字典。
    """
    # 确定文件前缀
    prefix = data_prefix or domain
    is_new_pipeline = data_prefix is not None
    
    data_loaders = {}
    
    # 正例对 DataLoader
    pos_name = f"{prefix}_train_positive_pairs.json" if is_new_pipeline else f"{prefix}_positive_pairs.json"
    pos_file = os.path.join(data_dir, pos_name)
    if os.path.exists(pos_file):
        pos_examples = load_positive_pairs(pos_file)
        data_loaders["positive_pairs"] = create_dataloader(pos_examples, batch_size)
    
    # 三元组 DataLoader
    tri_file = os.path.join(data_dir, f"{prefix}_triplets.json")
    if os.path.exists(tri_file):
        tri_examples = load_triplets(tri_file)
        data_loaders["triplets"] = create_dataloader(tri_examples, batch_size)
    
    # 余弦相似度 DataLoader
    cos_file = os.path.join(data_dir, f"{prefix}_cosine_pairs.json")
    if os.path.exists(cos_file):
        cos_examples = load_cosine_pairs(cos_file)
        data_loaders["cosine_pairs"] = create_dataloader(cos_examples, batch_size)
    
    print(f"\nTotal data loaders created: {len(data_loaders)}")
    for name, loader in data_loaders.items():
        print(f"  {name}: {len(loader.dataset)} samples, {len(loader)} batches")
    
    return data_loaders
