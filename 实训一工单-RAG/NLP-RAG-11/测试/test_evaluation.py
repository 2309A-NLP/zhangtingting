"""
NLP-RAG-11 评估模块单元测试
"""
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))


def test_import_evaluation():
    """测试评估模块导入"""
    try:
        from evaluation import evaluate_model, compare_models
        print("[PASS] evaluation 模块导入成功")
        return True
    except Exception as e:
        print(f"[FAIL] 导入失败: {e}")
        return False


def test_evaluate_before():
    """测试微调前评估"""
    try:
        from model_loading import load_model
        from evaluation import evaluate_model

        model = load_model('BAAI/bge-base-en-v1.5', device='cpu')
        metrics = evaluate_model(model, domain='psbc', name='Before FT')

        print(f"[INFO] Recall@1: {metrics.get('Recall@1', 'N/A')}")
        print(f"[INFO] MRR@10: {metrics.get('MRR@10', 'N/A')}")

        assert 'Recall@1' in metrics, "缺少 Recall@1"
        assert 'MRR@10' in metrics, "缺少 MRR@10"
        assert 0 <= metrics['Recall@1'] <= 1, "Recall@1 不在 [0,1] 范围"
        print("[PASS] 评估结果格式正确")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        return False


def test_evaluate_after():
    """测试微调后评估"""
    model_path = '/mnt/d/Desktop/NLP-RAG-11/models/finetuned_psbc'
    if not os.path.exists(model_path):
        print("[SKIP] 微调模型不存在")
        return None

    try:
        from model_loading import load_model
        from evaluation import evaluate_model

        model = load_model(model_path, device='cpu')
        metrics = evaluate_model(model, domain='psbc', name='After FT')

        print(f"[INFO] Recall@1: {metrics.get('Recall@1', 'N/A')}")
        print(f"[INFO] MRR@10: {metrics.get('MRR@10', 'N/A')}")

        assert metrics['Recall@1'] >= 0
        print("[PASS] 微调后评估完成")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        return False


def test_compare_models_improvement():
    """验证对比结果中 improvement 结构正确"""
    results_dir = '/mnt/d/Desktop/NLP-RAG-11/results'
    if not os.path.exists(results_dir):
        print("[SKIP] results 目录不存在")
        return None

    comp_files = [f for f in os.listdir(results_dir) if f.startswith('comparison_') and f.endswith('.json')]
    if not comp_files:
        print("[SKIP] 无对比 JSON 文件")
        return None

    comp_files.sort(reverse=True)
    with open(os.path.join(results_dir, comp_files[0])) as f:
        data = json.load(f)

    assert 'before' in data, "缺少 before"
    assert 'after' in data, "缺少 after"
    assert 'improvement' in data, "缺少 improvement"

    imp = data['improvement']
    for k, v in imp.items():
        assert 'before' in v, f"{k} 缺少 before"
        assert 'after' in v, f"{k} 缺少 after"
        assert 'absolute_change' in v, f"{k} 缺少 absolute_change"
        assert 'percentage_change' in v, f"{k} 缺少 percentage_change"

    print(f"[INFO] 对比文件: {comp_files[0]}")
    recall_imp = imp.get('Recall@1', {})
    if recall_imp:
        print(f"[INFO] Recall@1: {recall_imp['before']:.2%} → {recall_imp['after']:.2%} ({recall_imp['percentage_change']:+.2f}%)")

    print("[PASS] 对比结果结构完整")
    return True


if __name__ == '__main__':
    print("=" * 60)
    print("NLP-RAG-11 评估模块测试")
    print("=" * 60)

    tests = [
        ("导入测试", test_import_evaluation),
        ("微调前评估", test_evaluate_before),
        ("微调后评估", test_evaluate_after),
        ("对比结果验证", test_compare_models_improvement),
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
