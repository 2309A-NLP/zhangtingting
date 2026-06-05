#!/bin/bash
# NLP-RAG-11 部署脚本
# 将微调模型部署为可用的嵌入服务

set -e

PROJECT_DIR="/mnt/d/Desktop/NLP-RAG-11"
PYTHON="/home/ztt/miniconda3/envs/llamafactory/bin/python"
MODEL_DIR="$PROJECT_DIR/models/finetuned_psbc"
DEPLOY_DIR="$PROJECT_DIR/deploy"
WIN_PROJECT_DIR="D:\Desktop\NLP-RAG-11"
WIN_DEPLOY_DIR="D:\Desktop\NLP-RAG-11\deploy"

echo ""
echo "================================================================"
echo "  NLP-RAG-11 模型部署脚本"
echo "  版本: v1.0"
echo "  日期: $(date)"
echo "================================================================"

# 1. 检查微调模型
if [ ! -d "$MODEL_DIR" ]; then
    echo "[ERROR] 微调模型不存在: $MODEL_DIR"
    echo "请先运行训练"
    exit 1
fi
echo "[OK] 微调模型: $MODEL_DIR"

# 2. 检查 Python 环境
if [ ! -f "$PYTHON" ]; then
    echo "[ERROR] Python 环境不存在: $PYTHON"
    exit 1
fi
echo "[OK] Python 环境: $PYTHON"

# 3. 安装依赖
REQUIREMENTS="$PROJECT_DIR/requirements.txt"
if [ -f "$REQUIREMENTS" ]; then
    echo "[INFO] 检查依赖..."
    $PYTHON -m pip install -r "$REQUIREMENTS" -i https://pypi.tuna.tsinghua.edu.cn/simple --quiet 2>&1 | tail -1
    echo "[OK] 依赖安装完成"
fi

# 4. 创建部署目录
mkdir -p "$DEPLOY_DIR"
mkdir -p "$DEPLOY_DIR/models"
echo "[OK] 部署目录: $WIN_DEPLOY_DIR"

# 5. 复制模型文件
echo "[INFO] 复制模型文件..."
cp -r "$MODEL_DIR"/* "$DEPLOY_DIR/models/"
echo "[OK] 模型文件已复制"

# 6. 验证模型完整性
$PYTHON -c "
import sys
sys.path.insert(0, '$PROJECT_DIR')
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('$DEPLOY_DIR/models')
dim = model.get_sentence_embedding_dimension()
print(f'[OK] 模型验证成功: 维度={dim}, 最大序列长度={model.max_seq_length}')
"

# 7. 生成 API 入口脚本
cat > "$DEPLOY_DIR/embed.py" << 'PYEOF'
"""
NLP-RAG-11 嵌入模型 API

用法:
  python embed.py encode "文本"
  python embed.py search "查询"
  python embed.py serve
"""
import sys, os, json
import numpy as np

DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(DEPLOY_DIR, 'models')

def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_PATH)

def encode(texts, normalize=True):
    if isinstance(texts, str):
        texts = [texts]
    model = load_model()
    return model.encode(texts, normalize_embeddings=normalize)

def search(query, corpus, top_k=5):
    model = load_model()
    q_emb = model.encode([query], normalize_embeddings=True)[0]
    c_embs = model.encode(corpus, normalize_embeddings=True)
    scores = np.dot(c_embs, q_emb)
    top = np.argsort(scores)[-top_k:][::-1]
    return [{'index': int(i), 'text': corpus[i][:200], 'score': float(scores[i])} for i in top]

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python embed.py encode|search|serve")
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == 'encode':
        emb = encode(sys.argv[2])[0]
        print(f"嵌入维度: {len(emb)}\n前10个值: {emb[:10].tolist()}")
    elif cmd == 'search':
        query = sys.argv[2]
        corpus = [
            "中国邮政储蓄银行股份有限公司2019年年度报告",
            "本行持续优化风险治理架构，完善全面风险管理体系",
            "报告期内，本行实现营业收入2,768.09亿元，同比增长6.06%",
            "本行秉持普惠金融理念，服务社区、服务中小企业、服务'三农'",
            "不良贷款率0.86%，拨备覆盖率389.45%",
        ]
        for r in search(query, corpus):
            print(f"  [{r['score']:.4f}] {r['text']}")
    elif cmd == 'serve':
        try:
            from http.server import HTTPServer, BaseHTTPRequestHandler
            import urllib.parse
            model = load_model()
            print(f"[OK] 模型已加载，端口 8765")
            class H(BaseHTTPRequestHandler):
                def do_GET(self):
                    p = urllib.parse.urlparse(self.path)
                    q = urllib.parse.parse_qs(p.query)
                    if p.path == '/encode':
                        t = q.get('text', [''])[0]
                        if not t:
                            self.send_response(400); self.end_headers()
                            self.wfile.write(b'{"error":"missing text"}')
                            return
                        emb = model.encode([t], normalize_embeddings=True)[0]
                        self.send_response(200)
                        self.send_header('Content-Type','application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps({'dim':len(emb),'embedding':emb.tolist()}).encode())
                    else:
                        self.send_response(200)
                        self.send_header('Content-Type','text/html')
                        self.end_headers()
                        self.wfile.write(b'<h2>NLP-RAG-11 Embedding API</h2><form><input name="text" placeholder="输入文本" size="50"><button type="submit">编码</button></form>')
            HTTPServer(('0.0.0.0', 8765), H).serve_forever()
        except KeyboardInterrupt:
            print("\n[INFO] 服务已停止")
PYEOF
echo "[OK] API 脚本已生成: embed.py"

# 8. 生成部署说明
cat > "$DEPLOY_DIR/README.txt" << 'EOF'
NLP-RAG-11 模型部署说明
========================

部署目录结构:
  deploy/
  ├── models/          # 微调模型权重
  ├── embed.py         # API 入口
  └── README.txt       # 本文件

用法:
  1. 生成嵌入: python embed.py encode "你需要编码的文本"
  2. 检索文档: python embed.py search "查询内容"
  3. 启动服务: python embed.py serve
     -> 访问 http://localhost:8765/encode?text=你的文本

前置: Python 3.8+, sentence-transformers>=5.0, PyTorch>=2.0
  pip install sentence-transformers torch

验证: python embed.py encode "测试" -> 返回 768 维向量
EOF
echo "[OK] 部署说明已生成"

# 9. 生成 Windows 快捷入口
cat > "$PROJECT_DIR/deploy_service.bat" << 'BATEOF'
@echo off
chcp 65001 >nul
echo ================================================================
echo  NLP-RAG-11 嵌入服务启动脚本
echo  服务地址: http://localhost:8765
echo  按 Ctrl+C 停止服务
echo ================================================================
wsl bash -c "cd /mnt/d/Desktop/NLP-RAG-11 && /home/ztt/miniconda3/envs/llamafactory/bin/python /mnt/d/Desktop/NLP-RAG-11/deploy/embed.py serve"
pause
BATEOF
echo "[OK] Windows 启动脚本已生成: deploy_service.bat"

echo ""
echo "================================================================"
echo "  部署完成!"
echo ""
echo "  部署目录: $WIN_DEPLOY_DIR"
echo "    - models/          (微调模型权重)"
echo "    - embed.py         (API 调用脚本)"
echo "    - README.txt       (部署说明)"
echo ""
echo "  编码: python embed.py encode \"文本\""
echo "  检索: python embed.py search \"查询\""
echo "  服务: python embed.py serve"
echo "  快捷: deploy_service.bat"
echo "================================================================"
