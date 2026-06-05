# NLP-RAG-11 嵌入模型微调系统 - 设计文档

## 1. 项目概述

### 1.1 项目目标
对 BAAI/bge-base-en-v1.5 基础嵌入模型进行领域微调，使其在 PSBC（邮政储蓄银行）2019 年度报告文本的检索任务上获得可测量的效果提升。

### 1.2 技术栈

| 组件 | 技术选型 | 版本 |
| :--- | :--- | :--- |
| 基础模型 | BAAI/bge-base-en-v1.5 | v1.5 |
| 微调框架 | SentenceTransformers | 5.5.1 |
| 深度学习 | PyTorch | 2.8.0 (CUDA 12.8) |
| PDF 提取 | PyMuPDF (fitz) | - |
| 可视化 | Matplotlib | 3.x |

### 1.3 环境约束
- GPU: NVIDIA CUDA 12.8
- 操作系统: WSL (Ubuntu) + Windows
- 网络: WSL 无互联网，通过 Windows + hf-mirror.com 下载模型
- pip: 通过 Tsinghua 镜像加速

## 2. 系统架构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        NLP-RAG-11 Pipeline                          │
│                                                                     │
│  ┌──────────┐   ┌──────────────┐   ┌───────────┐   ┌───────────┐  │
│  │  PDF     │──▶│ Data Gen     │──▶│ Model     │──▶│ Training  │  │
│  │ Extract  │   │ (Section     │   │ Loading   │   │ (MNRL)    │  │
│  │ (PyMuPDF)│   │  Split + QA) │   │ (BGE)     │   │           │  │
│  └──────────┘   └──────────────┘   └───────────┘   └─────┬─────┘  │
│                                                           │        │
│  ┌──────────┐   ┌──────────────┐   ┌───────────┐         │        │
│  │ Charts   │◀──│ Comparison   │◀──│ Evaluate  │◀────────┘        │
│  │ (Matplot)│   │ (Before/After)│   │ (IR Eval) │                  │
│  └──────────┘   └──────────────┘   └───────────┘                  │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 模块职责

| 模块 | 文件 | 职责 |
| :--- | :--- | :--- |
| 数据生成 | `data_generation.py` | PDF 文本提取、分段、查询生成、负采样、训练/测试集划分 |
| 模型加载 | `model_loading.py` | BGE 模型加载、缓存管理、设备分配 |
| 损失函数 | `loss_functions.py` | 7 种损失函数的工厂模式封装 + MatryoshkaLoss |
| 训练引擎 | `training.py` | TrainingConfig、DataLoader 创建、SentenceTrainer 编排 |
| 评估引擎 | `evaluation.py` | InformationRetrievalEvaluator 封装、前后对比 |
| 可视化   | `visualization.py` | Matplotlib 图表生成 |
| 主控流程 | `main.py` | 9 步流水线编排、CLI 参数解析 |

## 3. 数据流设计

### 3.1 数据生成流程

```
PDF (421页) ──▶ PyMuPDF 逐页提取 ──▶ 按 TOC 章节分段
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
              训练章节 (13个)      测试章节 (3个)       跳过章节
              ..................... .................  .................
              经营业绩(6-17)        术语表(4-6)        股本变动(2)
              业务综述(23-60)       高管信息(18-22)     优先股(3)
              核心竞争力(60-67)     监事会报告(160-163)  公司治理(71-120)
              ...                                       ...
                                                                      
                    │                   │                              
                    ▼                   ▼                              
              段落拆分 (323段)     段落拆分 (23段)                     
                    │                   │                              
              5种查询策略            保留为测试集                       
              ├─ 原文摘录                                              
              ├─ 关键词替换                                            
              ├─ 摘要式                                                
              ├─ 追问式                                                
              └─ 段落融合                                              
                    │                                                 
                    1531 查询-文档对                                   
                    │                                                 
              跨章节负采样 (同域组内)                                  
                    │                                                 
              1433 训练对 / 98 测试对                                  
```

### 3.2 章节分组与负采样策略

将 16 个内容章节按主题分为 4 个域组，负样本从同组其他章节采样：

| 域组 | 包含章节 | 负样本策略 |
| :--- | :--- | :--- |
| **业务** (biz) | 业务综述、核心竞争力、经营情况讨论、募集资金 | 同组其他章节的段落 |
| **风险** (risk) | 风险与合规、重要事项 | 风险相关段落 |
| **财务** (fin) | 财务报表、财务指标 | 财务数据段落 |
| **治理** (gov) | 公司治理、股份变动、董事报告、社会责任 | 治理相关段落 |

## 4. 训练设计

### 4.1 损失函数选择

| 损失函数 | 适用场景 | 本项目结论 |
| :--- | :--- | :--- |
| **MNRL** | (查询, 文档, 分数) 三元组 | **最佳选择** - Recall@1 +4.08% |
| BatchHardTriplet | (句子, 类别标签) | 不兼容 - 数据格式不符 |
| CoSENT | (句子对, 相似度) | 梯度为 0，无学习效果 |
| CosineSimilarity | (句子对, 分数) | 训练后效果反而下降 (-8.16%) |

MNRL (MultipleNegativesRankingLoss) 原理：对每个正例对 (query, doc+)，将 batch 内其他文档作为负例 (doc1-, doc2-, ..., docN-)，最大化正例相似度同时最小化负例相似度。Batch 越大，负例越多，效果越好。

### 4.2 超参数配置

| 参数 | 值 | 说明 |
| :--- | :--- | :--- |
| 学习率 | 2e-5 | AdamW 标准学习率 |
| Batch Size | 16 | MNRL 负例数 = 15/ batch |
| Epochs | 10 | 900 步 (90 batch × 10) |
| Warmup | 500 步 | 余弦退火前预热 |
| Weight Decay | 0.01 | L2 正则化 |
| 优化器 | AdamW | 带权重衰减 |
| 调度器 | warmupcosine | 预热后余弦退火 |
| AMP | True | 混合精度加速 |

### 4.3 评估指标

使用 InformationRetrievalEvaluator：

- **Recall@K**：前 K 个检索结果中包含正确答案的比例
- **Precision@K**：前 K 个检索结果中正确结果的比例
- **MRR@K**：Mean Reciprocal Rank，正确答案排名的倒数均值

## 5. 模块设计详述

### 5.1 data_generation.py

```python
class DataGenerator:
    def __init__(self, pdf_path: str)
    def extract_text_by_page() -> List[str]
    def split_into_paragraphs() -> List[Paragraph]
    def generate_queries(paragraph) -> List[Query]
    def create_positive_pairs() -> List[Tuple[str, str, float]]
    def sample_negatives(pairs) -> List[Tuple[str, str, float]]
    def train_test_split() -> (train, test)
```

核心设计决策：
1. 从 TOC 硬编码页范围，避免正文中的小节编号干扰匹配
2. 5 种查询策略覆盖不同角度
3. 跨章节域组负采样，增加负例区分度
4. 按章节切分训练/测试集，防止数据泄露

### 5.2 loss_functions.py

工厂模式：

```python
def get_loss_function(model, loss_type: str, use_matryoshka: bool) -> nn.Module
```

支持的损失类型：cosine_similarity, contrastive, triplet, batch_hard_triplet, mnrl, mnrsl, cosent

MatryoshkaLoss 包装：在多维度上计算损失 (768/384/192/96/48)，强制模型在不同维度上都能保持语义。

### 5.3 training.py

```python
@dataclass
class TrainingConfig:
    num_epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 2e-5
    output_dir: str = './models'
    use_matryoshka: bool = False
```

基于 SentenceTransformers 的 SentenceTrainer 构建训练循环。

### 5.4 evaluation.py

```python
def evaluate_model(model, domain: str, name: str) -> Dict[str, float]
def compare_models(before, after, domain: str) -> Dict[str, Any]
```

封装 InformationRetrievalEvaluator，提供前后对比能力。

### 5.5 main.py

9 步流水线，通过 argparse 支持 CLI 参数化：

```bash
python src/main.py --domain psbc --data_prefix psbc --loss_type mnrl --num_epochs 10 --batch_size 16 --learning_rate 2e-5
```

## 6. 文件结构与路径约定

```
D:\Desktop\NLP-RAG-11\
├── src/
│   ├── main.py               # 主控流水线
│   ├── data_generation.py    # 数据生成
│   ├── model_loading.py      # 模型加载
│   ├── loss_functions.py     # 损失函数
│   ├── training.py           # 训练引擎
│   ├── evaluation.py         # 评估引擎
│   └── visualization.py      # 可视化
├── data/
│   ├── raw/2020-03-26.pdf    # 原始 PDF (421页)
│   └── processed/            # 生成的数据
│       ├── psbc_train_positive_pairs.json
│       ├── psbc_test_positive_pairs.json
│       ├── psbc_triplets.json
│       └── psbc_cosine_pairs.json
├── models/
│   └── finetuned_psbc/       # 微调后的模型权重
├── results/                  # 评估结果和图表
├── run.bat                   # Windows 双击运行入口
├── _gen_charts.py            # 图表生成脚本
└── requirements.txt          # 依赖清单
```

## 7. 设计决策记录

| 决策 | 选项 | 选择 | 理由 |
| :--- | :--- | :--- | :--- |
| 框架 | PyTorch / SentenceTransformers | SentenceTransformers | 内建训练循环、评估器、模型保存 |
| 损失函数 | MNRL / Triplet / CoSENT | MNRL | 最适合 (query, doc, score) 数据格式 |
| 评估器 | 自定义 / ST IR Eval | ST IR Eval | 标准化评估，支持 Recall/Precision/MRR |
| 数据格式 | (sentence, label) / (query, doc, score) | 后者 | 更自然表达检索关系 |
| 训练/测试划分 | 随机 / 按章节 | 按章节 | 防止数据泄露，评估更真实 |
| PDF 提取 | PyMuPDF / PDFPlumber / PyPDF2 | PyMuPDF | 速度最快，文本提取质量好 |
| 模型下载 | 直连 / hf-mirror / 缓存在 WSL | hf-mirror + robocopy | WSL 无网络，通过 Windows 中转 |
