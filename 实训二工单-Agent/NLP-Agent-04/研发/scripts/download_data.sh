#!/bin/bash
# =============================================
# 基金数据问答智能体系统 — 数据下载脚本
# =============================================
set -e

DATA_DIR="data/raw"
mkdir -p "$DATA_DIR"

echo "========================================"
echo "下载博金杯比赛数据集..."
echo "========================================"

# 方案：使用 git lfs 从 ModelScope 克隆
if ! command -v git-lfs &> /dev/null; then
    echo "⚠️  请先安装 git lfs"
    echo "  macOS: brew install git-lfs"
    echo "  Ubuntu: sudo apt install git-lfs"
    echo "  Windows: git lfs install"
    exit 1
fi

echo "正在下载数据集（约 1.5GB）..."
echo "来源: https://www.modelscope.cn/datasets/BJQW14B/bs_challenge_financial_14b_dataset"

cd "$DATA_DIR"
if [ ! -d "bs_challenge_financial_14b_dataset" ]; then
    git lfs clone https://www.modelscope.cn/datasets/BJQW14B/bs_challenge_financial_14b_dataset.git
else
    echo "数据集已存在，更新中..."
    cd bs_challenge_financial_14b_dataset && git pull
fi

echo ""
echo "✅ 下载完成！"
echo "数据位置: $DATA_DIR/bs_challenge_financial_14b_dataset/"
echo ""
echo "文件结构："
echo "  dataset/博金杯比赛数据.db  — 基金数据库 (1.46GB)"
echo "  pdf/                       — 招股书 PDF"
echo "  pdf_txt_file/              — 招股书 TXT"
echo "  question.jsonl             — 1000 道测试题"
