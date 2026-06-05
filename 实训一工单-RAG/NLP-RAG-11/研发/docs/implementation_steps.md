# Embeddings 模型微调 - 实现步骤与过程问题记录

## 项目信息

| 项目 | 详情 |
|------|------|
| 项目名称 | 人工智能NLP-RAG项目-11-Embeddings模型微调 |
| 工单编号 | 人工智能NLP-RAG项目-Embeddings模型微调任务 |
| 创建时间 | 2025年8月26日 |
| 创建人 | 王洪荣 |
| 预估工时 | 2 人日 |

## 技术选型

| 组件 | 选择 | 原因 |
|------|------|------|
| 基础模型 | BAAI/bge-base-en-v1.5 | 工单指定模型，MTEB 表现优异，资源友好 |
| 微调框架 | SentenceTransformers | 封装完善，支持多种损失函数和评估器 |
| 训练框架 | PyTorch + HuggingFace Transformers | 标准生态，兼容性好 |
| 损失函数 | CosineSimilarityLoss + TripletLoss + ContrastiveLoss + MatryoshkaLoss | 工单要求全部实现 |
| 评估指标 | Recall@k / Precision@k / MRR / NDCG | 标准检索评估指标 |
| 领域数据 | 法律领域 + 医疗领域 | 模拟专业领域术语和语义鸿沟场景 |

## 实现步骤

### 步骤 1: 需求分析
- 阅读并解析任务工单 PDF
- 确认核心目标：在专业数据上微调 Embedding 模型，提升 RAG 系统检索准确率
- 确认验收标准：微调后模型检索效果 > 微调前，有数据指标支撑

### 步骤 2: 数据集生成
- **输入**: 法律/医疗领域专业文档和术语
- **输出**: 三种格式的数据集 (正例对 / 三元组 / 带相似度分数的句子对)
- **方法**:
  - 构建 12 篇法律专业文档 (Rule 12, Statute of Frauds, 侵权法等)
  - 构建 15 个法律领域查询，与相关文档配对
  - 构建 3 篇医疗专业文档和 3 个医疗查询
  - 生成正例对、三元组(含负例)、余弦相似度对三种格式
  - 支持跨领域数据生成

### 步骤 3: 模型加载
- **模型**: BAAI/bge-base-en-v1.5 (768 维嵌入)
- **框架**: SentenceTransformers
- **特性**: 自动检测 CUDA/CPU，支持模型缓存

### 步骤 4: 训练数据加载
- 支持从 JSON 文件加载正例对、三元组、余弦相似度对
- 统一转换为 SentenceTransformer InputExample 格式
- 创建 PyTorch DataLoader

### 步骤 5: 损失函数实现
- **CosineSimilarityLoss**: 适用于带相似度分数的句子对
- **ContrastiveLoss**: 适用于正负句子对
- **TripletLoss**: 适用于 (锚点, 正例, 负例) 三元组
- **BatchHardTripletLoss**: 批量硬负例挖掘
- **MultipleNegativesRankingLoss**: 多负例排序损失 (MNR)
- **CoSENTLoss**: 余弦排序损失
- **MatryoshkaLoss**: 套娃损失 (自定义实现)，支持分层可截断嵌入
  - 多维度截断 (768/384/192/96/64)
  - 各维度独立计算损失后加权求和

### 步骤 6: 训练配置
- Batch size: 32
- Learning rate: 2e-5
- Epochs: 3 (可配置)
- Warmup steps: 500
- Optimizer: AdamW
- Scheduler: warmupcosine
- 混合精度训练 (AMP)

### 步骤 7: 微调前评估
- 使用 InformationRetrievalEvaluator
- 计算 Recall@k (k=1,3,5,10)
- 计算 Precision@k, MRR
- 建立基线指标

### 步骤 8: 执行微调
- 使用 SentenceTransformer.fit() API
- 支持训练中评估 (每 N 步)
- 自动保存最佳模型
- 记录训练日志

### 步骤 9: 微调后评估
- 重新加载微调后模型
- 使用相同评估数据/方法
- 计算所有指标

### 步骤 10: 结果对比
- 对比微调前后 Recall@k, Precision@k, MRR 指标
- 计算绝对提升和百分比提升
- 验证是否满足验收标准
- 输出汇总报告

## 过程问题与解决方案

### 问题 1: 参考文章无法访问
- **现象**: 知乎文章返回 403 错误
- **影响**: 无法获取直接的技术参考
- **解决**: 基于工单内容和对 Embedding 微调领域的理解自主实现

### 问题 2: PDF 文本提取
- **现象**: PyPDF2 提取中文 PDF 时单字分隔
- **影响**: 文本内容可读性差
- **解决**: 通过解析 PDF 元数据和逐页提取，结合上下文理解内容

### 问题 3: 专业领域数据获取
- **现象**: 需要专业领域数据但无真实数据源
- **解决**: 自主构建法律/医疗领域的合成数据集，包含真实专业术语和文档结构

### 问题 4: CPU 训练速度
- **现象**: 在 CPU 环境下训练可能较慢
- **解决**: 支持 CUDA 加速，如无 GPU 可降低 epoch 数或使用更小批次

## 验收结果验证方法

1. 运行 `python src/main.py` 执行完整流程
2. 查看 `results/comparison_{timestamp}.json` 获取对比数据
3. 确认所有指标均有提升或持平
4. 查看 `results/summary_{timestamp}.json` 获取验收概要

## 附录

### 项目结构

```
NLP-RAG-11/
├── README.md                    # 项目需求分析文档
├── docs/
│   └── implementation_steps.md   # 实现步骤与过程问题记录
├── data/
│   ├── raw/                     # 原始数据
│   └── processed/               # 处理后数据集 (自动生成)
├── src/
│   ├── data_generation.py       # 数据集生成
│   ├── model_loading.py         # 模型与数据集加载
│   ├── loss_functions.py        # 损失函数 (含 Matryoshka)
│   ├── training.py              # 训练配置与流程
│   ├── evaluation.py            # 评估器与对比
│   └── main.py                  # 主运行脚本
├── models/                      # 微调后模型保存位置
├── results/                     # 评估结果与对比报告
└── requirements.txt            # Python 依赖
```

### 数据格式示例

**正例对 (positive_pairs.json)**
```json
[
  ["What is a motion to dismiss?", "Rule 12 governs defenses and objections...", 0.9],
  ...
]
```

**三元组 (triplets.json)**
```json
[
  ["What is a motion to dismiss?", "Rule 12 governs...", "The Fourth Amendment protects..."],
  ...
]
```

### 模型信息

- 基础模型: BAAI/bge-base-en-v1.5
- 嵌入维度: 768
- 最大序列长度: 512 (可调整)
- 参数量: ~110M
- 适用场景: 语义搜索、RAG 检索、文本聚类
