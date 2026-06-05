"""
NLP-RAG-11 训练引擎模块单元测试
"""
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))


def test_import_training():
    """测试训练模块导入"""
    try:
        from training import TrainingConfig, create_trainer, save_training_config
        print("[PASS] training 模块导入成功")
        return True
    except Exception as e:
        print(f"[FAIL] 导入失败: {e}")
        return False


def test_training_config_defaults():
    """测试 TrainingConfig 默认值"""
    try:
        from training import TrainingConfig
        config = TrainingConfig()
        print(f"[INFO] 默认 num_epochs: {config.num_epochs}")
        print(f"[INFO] 默认 batch_size: {config.batch_size}")
        print(f"[INFO] 默认 learning_rate: {config.learning_rate}")
        print(f"[INFO] 默认 output_dir: {config.output_dir}")
        assert config.num_epochs == 3
        assert config.batch_size == 32
        assert config.learning_rate == 2e-5
        print("[PASS] 默认参数正确")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        return False


def test_data_loader_creation():
    """测试 DataLoader 创建"""
    try:
        from sentence_transformers import SentenceTransformer
        from training import create_train_dataloader
        from loss_functions import get_loss_function

        train_path = '/mnt/d/Desktop/NLP-RAG-11/data/processed/psbc_train_positive_pairs.json'
        if not os.path.exists(train_path):
            print("[SKIP] 训练 JSON 不存在")
            return None

        with open(train_path) as f:
            data = json.load(f)

        model = SentenceTransformer('BAAI/bge-base-en-v1.5')
        loss_fn = get_loss_function(model, 'mnrl', use_matryoshka=False)
        loader = create_train_dataloader(data, model, loss_fn, batch_size=16)

        print(f"[INFO] DataLoader 批次数: {len(loader)}")
        assert len(loader) > 0, "DataLoader 为空"
        print("[PASS] DataLoader 创建成功")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        return False


def test_save_training_config():
    """测试训练配置保存"""
    import tempfile
    try:
        from training import TrainingConfig, save_training_config
        config = TrainingConfig(num_epochs=5, batch_size=16, learning_rate=1e-5)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            save_training_config(config, f.name)
            fname = f.name

        with open(fname) as f:
            saved = json.load(f)

        os.unlink(fname)

        assert saved['num_epochs'] == 5
        assert saved['batch_size'] == 16
        assert saved['learning_rate'] == 1e-5
        print("[PASS] 配置保存与读取正确")
        return True
    except Exception as e:
        print(f"[FAIL] {e}")
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("NLP-RAG-11 训练引擎模块测试")
    print("=" * 60)

    tests = [
        ("导入测试", test_import_training),
        ("配置默认值", test_training_config_defaults),
        ("DataLoader", test_data_loader_creation),
        ("配置保存", test_save_training_config),
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
