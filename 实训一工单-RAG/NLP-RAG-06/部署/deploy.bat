@echo off
chcp 65001 >nul
rem ==============================================================================
rem NLP-RAG-04 Windows 部署脚本
rem 适用于在 Windows cmd.exe 中操作 Docker 和模型下载
rem ==============================================================================

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%.."

cd /d "%PROJECT_DIR%"

echo.
echo ============================================
echo  NLP-RAG-04 Windows 部署助手
echo ============================================
echo.

rem ---- 1. 检查 Docker Desktop ----
docker ps >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [错误] Docker Desktop 未运行，请先启动 Docker Desktop
    pause
    exit /b 1
)
echo [OK] Docker Desktop 运行中

rem ---- 2. 启动 Docker 服务 ----
echo [INFO] 启动 Docker 基础设施...

echo [INFO] 启动 etcd...
docker compose up -d etcd
timeout /t 3 /nobreak >nul

echo [INFO] 启动 MinIO...
docker compose up -d minio
timeout /t 2 /nobreak >nul

echo [INFO] 启动 Milvus...
docker compose up -d milvus

echo [INFO] 等待 Milvus 就绪（最多 60 秒）...
set "milvus_ready=0"
for /l %%i in (1,1,30) do (
    curl -sf http://localhost:9092/health >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        set "milvus_ready=1"
        goto :milvus_ok
    )
    timeout /t 2 /nobreak >nul
)
:milvus_ok
if "!milvus_ready!"=="1" (
    echo [OK] Milvus 就绪
) else (
    echo [警告] Milvus 未在预期内就绪，继续启动其他服务
)

echo [INFO] 启动 Attu...
docker compose up -d attu
timeout /t 2 /nobreak >nul

echo [INFO] 启动 vLLM...
docker compose up -d vllm

echo.
echo [OK] 所有 Docker 服务已启动
docker compose ps

rem ---- 3. 提示后续操作 ----
echo.
echo ============================================
echo  部署完成！请在 WSL 终端中继续：
echo ============================================
echo.
echo  1. 激活 conda 环境并安装依赖：
echo     conda activate nlp-rag
echo     pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
echo.
echo  2. 启动 FastAPI 后端：
echo     uvicorn app.main:app --host 0.0.0.0 --port 8000
echo.
echo  3. 启动 Gradio 前端：
echo     python frontend/gradio_app.py
echo.
echo  4. 接入 PDF 数据：
echo     python test/batch_fill_rag_answers.py --ingest
echo.
echo  5. 访问服务：
echo     FastAPI:  http://localhost:8000
echo     Gradio:   http://localhost:7860
echo     Attu:     http://localhost:3001
echo.
echo  6. 停止服务：
echo     docker compose down
echo.

pause