"""
评估器与微调前后评估模块

实现模型评估功能，包括：
1. 检索评估 (Recall@k, MRR, Precision@k)
2. 语义相似度评估
3. 微调前后对比
4. 结果可视化
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    EmbeddingSimilarityEvaluator,
    TripletEvaluator,
    SentenceEvaluator,
)
from sklearn.metrics.pairwise import cosine_similarity
import torch


# ==================== 评估准备 ====================

def prepare_retrieval_data(
    domain: str = "legal",
    data_dir: str = None,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, List[str]]]:
    """
    准备检索评估数据
    
    Returns:
        (corpus, queries, relevant_docs)
        - corpus: {doc_id: doc_text}
        - queries: {query_id: query_text}
        - relevant_docs: {query_id: [list_of_relevant_doc_ids]}
    """
    import sys, os
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    
    # For PSBC domain: load from generated data files
    if domain == "psbc":
        _data_dir = data_dir or os.path.join(_root, "data", "processed")
        pos_file = os.path.join(_data_dir, "psbc_test_positive_pairs.json")
        if os.path.exists(pos_file):
            with open(pos_file, "r", encoding="utf-8") as f:
                pairs = json.load(f)
            # Build corpus from unique documents, queries from unique queries
            corpus = {}
            queries = {}
            relevant = {}
            doc_counter = 0
            query_counter = 0
            seen_docs = {}
            for q, d, s in pairs:
                # Assign doc IDs
                if d not in seen_docs:
                    doc_id = f"doc_{doc_counter}"
                    seen_docs[d] = doc_id
                    corpus[doc_id] = d
                    doc_counter += 1
                else:
                    doc_id = seen_docs[d]
                # Assign query IDs
                qid = f"query_{query_counter}"
                queries[qid] = q
                query_counter += 1
                if qid not in relevant:
                    relevant[qid] = []
                relevant[qid].append(doc_id)
            return corpus, queries, relevant
    
    # Legacy domains: use hardcoded data
    from src.data_generation import (
        LEGAL_DOCUMENTS, LEGAL_QUERIES, QUERY_DOC_MAP,
        MEDICAL_DOCUMENTS, MEDICAL_QUERIES, MEDICAL_QUERY_DOC_MAP,
    )
    
    if domain == "legal":
        docs = LEGAL_DOCUMENTS
        queries_list = LEGAL_QUERIES
        query_map = QUERY_DOC_MAP
    else:
        docs = MEDICAL_DOCUMENTS
        queries_list = MEDICAL_QUERIES
        query_map = MEDICAL_QUERY_DOC_MAP
    
    corpus = {}
    for i, doc in enumerate(docs):
        corpus[f"doc_{i}"] = doc["content"]
    
    query_dict = {}
    relevant = {}
    for q_idx, doc_idx in query_map.items():
        qid = f"query_{q_idx}"
        query_dict[qid] = queries_list[q_idx]
        relevant[qid] = [f"doc_{doc_idx}"]
    
    return corpus, query_dict, relevant


def create_retrieval_evaluator(
    domain: str = "legal",
    name: str = "legal-retrieval",
    batch_size: int = 32,
) -> InformationRetrievalEvaluator:
    """
    创建信息检索评估器
    """
    corpus, queries, relevant_docs = prepare_retrieval_data(domain)
    
    # 使用所有文档作为语料库
    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=name,
        batch_size=batch_size,
        show_progress_bar=True,
        # 评估多个 k 值
        accuracy=True,  # Hits@k
        precision=True,  # Precision@k
        recall=True,     # Recall@k
        mrr=True,        # MRR
        ndcg=True,       # NDCG
        map=True,        # MAP
        score_function="cosine",
    )
    
    return evaluator


def create_similarity_evaluator(
    pairs_file: str,
    name: str = "similarity",
    batch_size: int = 32,
) -> Optional[EmbeddingSimilarityEvaluator]:
    """
    创建语义相似度评估器
    
    需要包含 (sentence1, sentence2, score) 格式的评估数据
    """
    if not os.path.exists(pairs_file):
        print(f"Warning: pairs file not found: {pairs_file}")
        return None
    
    with open(pairs_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    sentences1 = []
    sentences2 = []
    scores = []
    
    for item in data:
        if isinstance(item, list) and len(item) >= 3:
            sentences1.append(str(item[0]))
            sentences2.append(str(item[1]))
            scores.append(float(item[2]))
    
    evaluator = EmbeddingSimilarityEvaluator(
        sentences1=sentences1,
        sentences2=sentences2,
        scores=scores,
        batch_size=batch_size,
        name=name,
        show_progress_bar=True,
    )
    
    return evaluator


# ==================== 手动评估 ====================

def compute_retrieval_metrics(
    model: SentenceTransformer,
    corpus: Dict[str, str],
    queries: Dict[str, str],
    relevant_docs: Dict[str, List[str]],
    k_values: List[int] = [1, 3, 5, 10],
) -> Dict[str, float]:
    """
    手动计算检索评估指标
    """
    # 编码所有文档和查询
    doc_ids = list(corpus.keys())
    doc_texts = [corpus[did] for did in doc_ids]
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    
    print(f"Encoding {len(doc_texts)} documents...")
    doc_embeddings = model.encode(doc_texts, convert_to_tensor=True, show_progress_bar=True)
    print(f"Encoding {len(query_texts)} queries...")
    query_embeddings = model.encode(query_texts, convert_to_tensor=True, show_progress_bar=True)
    
    # 计算相似度矩阵
    doc_embeddings_np = doc_embeddings.cpu().numpy() if torch.is_tensor(doc_embeddings) else doc_embeddings
    query_embeddings_np = query_embeddings.cpu().numpy() if torch.is_tensor(query_embeddings) else query_embeddings
    
    similarities = cosine_similarity(query_embeddings_np, doc_embeddings_np)
    
    metrics = {}
    for k in k_values:
        recall_list = []
        mrr_list = []
        precision_list = []
        
        for qi, qid in enumerate(query_ids):
            if qid not in relevant_docs:
                continue
            
            relevant_set = set(relevant_docs[qid])
            
            # 获取 top-k 文档
            top_k_indices = np.argsort(similarities[qi])[::-1][:k]
            top_k_docs = set([doc_ids[idx] for idx in top_k_indices])
            
            # Recall@k
            if relevant_set:
                hits = len(relevant_set & top_k_docs)
                recall_list.append(hits / len(relevant_set))
            
            # Precision@k
            precision_list.append(len(relevant_set & top_k_docs) / k)
            
            # MRR
            for rank, idx in enumerate(top_k_indices):
                if doc_ids[idx] in relevant_set:
                    mrr_list.append(1.0 / (rank + 1))
                    break
            else:
                mrr_list.append(0.0)
        
        metrics[f"Recall@{k}"] = np.mean(recall_list) if recall_list else 0.0
        metrics[f"Precision@{k}"] = np.mean(precision_list) if precision_list else 0.0
        metrics[f"MRR@{k}"] = np.mean(mrr_list) if mrr_list else 0.0
    
    return metrics


def evaluate_model(
    model: SentenceTransformer,
    domain: str = "legal",
    name: str = "",
    k_values: List[int] = [1, 3, 5, 10],
) -> Dict[str, float]:
    """
    评估模型的检索性能
    """
    print(f"\n{'=' * 60}")
    print(f"Evaluating Model: {name}")
    print(f"{'=' * 60}")
    
    corpus, queries, relevant_docs = prepare_retrieval_data(domain)
    
    print(f"  Domain: {domain}")
    print(f"  Corpus size: {len(corpus)}")
    print(f"  Queries: {len(queries)}")
    print(f"  Relevant pairs: {sum(len(v) for v in relevant_docs.values())}")
    
    metrics = compute_retrieval_metrics(
        model=model,
        corpus=corpus,
        queries=queries,
        relevant_docs=relevant_docs,
        k_values=k_values,
    )
    
    print(f"\n  Results ({name}):")
    for metric, value in metrics.items():
        print(f"    {metric}: {value:.4f}")
    print(f"{'=' * 60}")
    
    return metrics


def compare_models(
    model_before: SentenceTransformer,
    model_after: SentenceTransformer,
    domain: str = "legal",
    save_path: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    对比微调前后模型的性能
    """
    print(f"\n{'=' * 60}")
    print("Comparison: Before vs After Fine-tuning")
    print(f"{'=' * 60}")
    
    metrics_before = evaluate_model(model_before, domain, name="Before Fine-tuning")
    metrics_after = evaluate_model(model_after, domain, name="After Fine-tuning")
    
    comparison = {
        "before": metrics_before,
        "after": metrics_after,
        "improvement": {}
    }
    
    print(f"\n{'=' * 60}")
    print("Improvement Summary:")
    print(f"{'=' * 60}")
    
    for metric in metrics_before:
        before = metrics_before[metric]
        after = metrics_after[metric]
        improvement = after - before
        pct = (improvement / before * 100) if before > 0 else float('inf')
        comparison["improvement"][metric] = {
            "before": before,
            "after": after,
            "absolute_change": improvement,
            "percentage_change": pct,
        }
        arrow = "" if improvement >= 0 else ""
        print(f"  {metric}: {before:.4f}  {arrow}  {after:.4f}  ({pct:+.2f}%)")
    
    print(f"{'=' * 60}")
    
    # 保存结果
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            # Convert numpy values to float for JSON serialization
            def convert(obj):
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                return obj
            json.dump(convert(comparison), f, ensure_ascii=False, indent=2)
        print(f"Comparison results saved to {save_path}")
    
    return comparison


def evaluate_at_dimensions(
    model: SentenceTransformer,
    dimensions: List[int],
    domain: str = "legal",
) -> Dict[int, Dict[str, float]]:
    """
    评估模型在不同嵌入维度下的性能 (Matryoshka 评估)
    """
    corpus, queries, relevant_docs = prepare_retrieval_data(domain)
    
    doc_ids = list(corpus.keys())
    doc_texts = [corpus[did] for did in doc_ids]
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    
    print(f"\n{'=' * 60}")
    print("Matryoshka Dimensionality Evaluation")
    print(f"{'=' * 60}")
    
    # 获取完整嵌入
    doc_embeddings = model.encode(doc_texts, convert_to_tensor=True, show_progress_bar=True)
    query_embeddings = model.encode(query_texts, convert_to_tensor=True, show_progress_bar=True)
    
    doc_np = doc_embeddings.cpu().numpy() if torch.is_tensor(doc_embeddings) else doc_embeddings
    query_np = query_embeddings.cpu().numpy() if torch.is_tensor(query_embeddings) else query_embeddings
    
    results = {}
    for dim in sorted(dimensions, reverse=True):
        doc_slice = doc_np[:, :dim]
        query_slice = query_np[:, :dim]
        
        similarities = cosine_similarity(query_slice, doc_slice)
        
        recall_ks = []
        for qi, qid in enumerate(query_ids):
            if qid not in relevant_docs:
                continue
            relevant_set = set(relevant_docs[qid])
            top_k_indices = np.argsort(similarities[qi])[::-1][:5]
            top_k_docs = set([doc_ids[idx] for idx in top_k_indices])
            if relevant_set:
                recall_ks.append(len(relevant_set & top_k_docs) / len(relevant_set))
        
        recall = np.mean(recall_ks) if recall_ks else 0.0
        results[dim] = {"Recall@5": recall}
        print(f"  Dim {dim:4d}: Recall@5 = {recall:.4f}")
    
    print(f"{'=' * 60}")
    return results


if __name__ == "__main__":
    print("Evaluation Module Test")
    print("=" * 60)
    print("Available functions:")
    print("  - evaluate_model(): Evaluate retrieval performance")
    print("  - compare_models(): Compare before/after fine-tuning")
    print("  - evaluate_at_dimensions(): Matryoshka dimension analysis")
    print("  - create_retrieval_evaluator(): Create IR evaluator")
    print("=" * 60)
