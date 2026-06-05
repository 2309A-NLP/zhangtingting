"""
NLP-RAG-11 损失函数模块单元测试
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))


def test_import_loss_functions():
    """测试损失函数模块导入"""
    try:
        from loss_functions import get_loss_function, LOSS_TYPE_MAP
        print("[PASS] loss_functions 模块导入成功")
        print(f"[INFO] 支持 {len(LOSS_TYPE_MAP)} 种损失函数: {list(LOSS_TYPE_MAP.keys())}")
        return True
    except Exception as e:
        print(f"[FAIL] 导入失败: {e}")
        return False


def test_all_loss_types():
    """测试所有损失函数类型能否实例化"""
    try:
        from sentence_transformers import SentenceTransformer
        from loss_functions import get_loss_function, LOSS_TYPE_MAP

        model = SentenceTransformer('BAAI/bge-base-en-v1.5')

        for loss_type in LOSS_TYPE_MAP.keys():
            try:
                loss_fn = get_loss_function(model, loss_type, use_matryoshka=False)
                assert loss_fn is not None, f"{loss_type} 返回 None"
                print(f"  [PASS] {loss_type}")
            except Exception as e:
                print(f"  [FAIL] {loss_type}: {e}")
                return False

        print("[PASS] 所有损失函数类型均可实例化")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        return False


def test_matryoshka_wrapper():
    """测试 MatryoshkaLoss 包装"""
    try:
        from sentence_transformers import SentenceTransformer
        from loss_functions import get_loss_function

        model = SentenceTransformer('BAAI/bge-base-en-v1.5')
        loss_fn = get_loss_function(model, 'mnrl', use_matryoshka=True)
        print(f"[INFO] MatryoshkaLoss 类型: {type(loss_fn).__name__}")
        assert 'Matryoshka' in type(loss_fn).__name__, "应该是 MatryoshkaLoss"
        print("[PASS] MatryoshkaLoss 包装正确")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        return False


def test_loss_forward():
    """测试损失函数前向传播"""
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        from loss_functions import get_loss_function

        model = SentenceTransformer('BAAI/bge-base-en-v1.5')
        loss_fn = get_loss_function(model, 'mnrl', use_matryoshka=False)

        # 创建模拟数据
        features = model.tokenize(["query text", "doc1 text", "doc2 text"])
        labels = torch.tensor([0, 1, 1])

        loss = loss_fn(features, labels)
        print(f"[INFO] 损失值: {loss.item():.4f}")
        assert isinstance(loss, torch.Tensor), "损失应为 Tensor"
        assert loss.item() > 0, "损失应为正数"
        print("[PASS] 前向传播正常")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("NLP-RAG-11 损失函数模块测试")
    print("=" * 60)

    tests = [
        ("导入测试", test_import_loss_functions),
        ("全部损失类型", test_all_loss_types),
        ("Matryoshka 包装", test_matryoshka_wrapper),
        ("前向传播", test_loss_forward),
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
