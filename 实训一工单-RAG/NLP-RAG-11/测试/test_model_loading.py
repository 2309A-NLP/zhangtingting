"""
NLP-RAG-11 模型加载模块单元测试
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))


def test_import_model_loading():
    """测试模型加载模块能否正常导入"""
    try:
        from model_loading import load_model
        print("[PASS] model_loading 模块导入成功")
        return True
    except Exception as e:
        print(f"[FAIL] 导入失败: {e}")
        return False


def test_load_base_model():
    """测试加载 BGE 基础模型"""
    try:
        from model_loading import load_model
        model = load_model('BAAI/bge-base-en-v1.5', device='cpu')
        dim = model.get_sentence_embedding_dimension()
        max_seq = model.max_seq_length
        print(f"[INFO] 嵌入维度: {dim}")
        print(f"[INFO] 最大序列长度: {max_seq}")
        assert dim == 768, f"维度应为 768，实际: {dim}"
        assert max_seq == 512, f"序列长度应为 512，实际: {max_seq}"
        print("[PASS] 模型加载正确，参数匹配预期")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        return False


def test_load_finetuned_model():
    """测试加载微调后模型"""
    model_path = '/mnt/d/Desktop/NLP-RAG-11/models/finetuned_psbc'
    if not os.path.exists(model_path):
        print("[SKIP] 微调模型目录不存在")
        return None

    try:
        from model_loading import load_model
        model = load_model(model_path, device='cpu')
        dim = model.get_sentence_embedding_dimension()
        print(f"[INFO] 微调模型维度: {dim}")
        assert dim == 768, f"维度应为 768"
        print("[PASS] 微调模型加载成功")

        # 测试推理
        emb = model.encode("test sentence", normalize_embeddings=True)
        assert len(emb) == 768, f"embedding 长度应为 768"
        assert abs(sum(emb) - 1.0) < 0.1, "L2 归一化验证失败"
        print("[PASS] 推理正常，归一化正确")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        return False


def test_embedding_consistency():
    """测试相同输入产生相同输出"""
    try:
        from model_loading import load_model
        model = load_model('BAAI/bge-base-en-v1.5', device='cpu')
        emb1 = model.encode("Postal Savings Bank of China", normalize_embeddings=True)
        emb2 = model.encode("Postal Savings Bank of China", normalize_embeddings=True)
        diff = sum(abs(a - b) for a, b in zip(emb1, emb2))
        assert diff < 1e-5, f"两次编码结果不一致: diff={diff}"
        print("[PASS] 推理结果一致，无随机性")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("NLP-RAG-11 模型加载模块测试")
    print("=" * 60)

    tests = [
        ("导入测试", test_import_model_loading),
        ("基础模型加载", test_load_base_model),
        ("微调模型加载", test_load_finetuned_model),
        ("推理一致性", test_embedding_consistency),
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
