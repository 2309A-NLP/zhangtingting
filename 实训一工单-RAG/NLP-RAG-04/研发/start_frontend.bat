@echo off
setlocal

cd /d "%~dp0"

set "PROJECT_ROOT=%CD%"
set "NLP_RAG_PYTHON=D:\anaconda2024\envs\nlp-rag\python.exe"

if /I "%~1"=="check" goto check

call :validate || exit /b 1

echo Starting frontend ...
"%NLP_RAG_PYTHON%" frontend\gradio_app.py
exit /b %errorlevel%

:check
call :validate || exit /b 1
echo Frontend script check passed.
echo Command: "%NLP_RAG_PYTHON%" frontend\gradio_app.py
exit /b 0

:validate
if not exist "%NLP_RAG_PYTHON%" (
    echo Failed to find Python interpreter: %NLP_RAG_PYTHON%
    exit /b 1
)
if not exist "%PROJECT_ROOT%\frontend\gradio_app.py" (
    echo Missing frontend entry: %PROJECT_ROOT%\frontend\gradio_app.py
    exit /b 1
)
"%NLP_RAG_PYTHON%" -c "import gradio, requests" >nul 2>nul
if errorlevel 1 (
    echo Frontend dependency check failed. Verify the nlp-rag environment.
    exit /b 1
)
exit /b 0
