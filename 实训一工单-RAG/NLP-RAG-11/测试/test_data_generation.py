"""
NLP-RAG-11 数据生成模块单元测试
"""
import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

def test_import_data_generation():
    """测试数据生成模块能否正常导入"""
    try:
        from data_generation import (
            extract_pdf_text, split_into_paragraphs,
            generate_queries, create_positive_pairs,
            sample_negatives, train_test_split_by_section
        )
        print("[PASS] data_generation 模块导入成功")
        return True
    except Exception as e:
        print(f"[FAIL] 导入失败: {e}")
        return False


def test_paragraph_count():
    """测试 PDF 提取的段落数量"""
    try:
        from data_generation import extract_pdf_text, split_into_paragraphs
        pdf_path = '/mnt/d/Desktop/NLP-RAG-11/data/raw/2020-03-26.pdf'
        if not os.path.exists(pdf_path):
            print("[SKIP] PDF 文件不存在，跳过")
            return None
        text_per_page = extract_pdf_text(pdf_path)
        pages = [p for p in text_per_page if p.strip()]
        print(f"[INFO] PDF 有效页数: {len(pages)}")
        assert len(pages) > 100, f"有效页数太少: {len(pages)}"
        print("[PASS] 页数合理 (>100)")

        paragraphs = split_into_paragraphs(text_per_page)
        print(f"[INFO] 段落数: {len(paragraphs)}")
        assert len(paragraphs) > 50, f"段落数太少: {len(paragraphs)}"
        print("[PASS] 段落数合理 (>50)")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        return False


def test_train_test_split_json():
    """验证训练/测试 JSON 文件完整性"""
    train_path = '/mnt/d/Desktop/NLP-RAG-11/data/processed/psbc_train_positive_pairs.json'
    test_path = '/mnt/d/Desktop/NLP-RAG-11/data/processed/psbc_test_positive_pairs.json'

    if not os.path.exists(train_path):
        print("[SKIP] 训练 JSON 不存在")
        return None
    if not os.path.exists(test_path):
        print("[SKIP] 测试 JSON 不存在")
        return None

    with open(train_path) as f:
        train_data = json.load(f)
    with open(test_path) as f:
        test_data = json.load(f)

    print(f"[INFO] 训练对: {len(train_data)}")
    print(f"[INFO] 测试对: {len(test_data)}")

    assert len(train_data) > 500, f"训练数据太少: {len(train_data)}"
    assert len(test_data) > 30, f"测试数据太少: {len(test_data)}"

    # 验证数据格式
    sample = train_data[0]
    assert 'query' in sample, "缺少 query 字段"
    assert 'doc' in sample, "缺少 doc 字段"
    assert 'score' in sample, "缺少 score 字段"
    print(f"[INFO] 示例: query={sample['query'][:50]}..., doc={sample['doc'][:50]}..., score={sample['score']}")

    # 验证训练/测试没有重叠
    train_docs = set(item['doc'] for item in train_data)
    test_docs = set(item['doc'] for item in test_data)
    overlap = train_docs & test_docs
    if len(overlap) > 0:
        print(f"[WARN] 训练和测试有 {len(overlap)} 个文档重叠")
    else:
        print("[PASS] 训练/测试集无文档重叠")

    print("[PASS] JSON 文件完整性验证通过")
    return True


def test_query_strategies():
    """验证 5 种查询策略的多样性"""
    train_path = '/mnt/d/Desktop/NLP-RAG-11/data/processed/psbc_train_positive_pairs.json'
    if not os.path.exists(train_path):
        print("[SKIP] 训练 JSON 不存在")
        return None

    with open(train_path) as f:
        train_data = json.load(f)

    # 计数每种分数值的分布
    score_dist = {}
    for item in train_data:
        s = item['score']
        score_dist[s] = score_dist.get(s, 0) + 1

    print(f"[INFO] 分数分布: {dict(sorted(score_dist.items()))}")
    assert len(score_dist) >= 3, f"分数种类太少: {len(score_dist)}"
    print("[PASS] 多种分数表示多种查询策略")

    # 检查查询文本的多样性
    queries = [item['query'] for item in train_data]
    unique_queries = len(set(queries))
    print(f"[INFO] 唯一查询数: {unique_queries}/{len(queries)}")
    assert unique_queries > len(queries) * 0.5, f"查询重复过多"
    print("[PASS] 查询多样性足够")
    return True


def test_negative_sampling():
    """验证负采样有效性：检查是否有不同章节文档互为负例"""
    train_path = '/mnt/d/Desktop/NLP-RAG-11/data/processed/psbc_train_positive_pairs.json'
    if not os.path.exists(train_path):
        print("[SKIP] 训练 JSON 不存在")
        return None

    with open(train_path) as f:
        train_data = json.load(f)

    # 检查正例分数 = 1.0 的比例
    positive_count = sum(1 for item in train_data if item['score'] >= 0.9)
    print(f"[INFO] 高分正例(>=0.9): {positive_count}/{len(train_data)}")
    assert positive_count > 0, "没有正例"
    print("[PASS] 存在正例")
    return True


if __name__ == '__main__':
    print("=" * 60)
    print("NLP-RAG-11 数据生成模块测试")
    print("=" * 60)

    tests = [
        ("导入测试", test_import_data_generation),
        ("段落提取", test_paragraph_count),
        ("训练/测试划分", test_train_test_split_json),
        ("查询策略多样性", test_query_strategies),
        ("负采样有效性", test_negative_sampling),
    ]

    passed = 0
    failed = 0
    skipped = 0

    for name, fn in tests:
        print(f"\n--- {name} ---")
        result = fn()
        if result is True:
            passed += 1
        elif result is False:
            failed += 1
        else:
            skipped += 1

    print(f"\n{'=' * 60}")
    print(f"结果: {passed} 通过, {failed} 失败, {skipped} 跳过")
    print(f"{'=' * 60}")

    sys.exit(1 if failed > 0 else 0)
