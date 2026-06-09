@echo off
setlocal

cd /d "%~dp0"

set "PROJECT_ROOT=%CD%"
set "NLP_RAG_PYTHON=D:\anaconda2024\envs\nlp-rag\python.exe"
set "PDF_PARSER_PYTHON=D:\anaconda2024\envs\pdf-parser\python.exe"
set "PDF_PARSER_BACKEND=parse2"
if /I "%~1"=="check" goto check

call :validate || exit /b 1

for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo Stopping existing process on port 8000: PID %%p
    taskkill /PID %%p /F >nul 2>nul
)

echo Starting backend on http://127.0.0.1:8000 ...
echo Query mode: unified default corpus routing
"%NLP_RAG_PYTHON%" -m uvicorn app.main:app --host 0.0.0.0 --port 8000
exit /b %errorlevel%

:check
call :validate || exit /b 1
echo Backend script check passed.
echo Query mode: unified default corpus routing
echo Command: "%NLP_RAG_PYTHON%" -m uvicorn app.main:app --host 0.0.0.0 --port 8000
exit /b 0

:validate
if not exist "%NLP_RAG_PYTHON%" (
    echo Failed to find Python interpreter: %NLP_RAG_PYTHON%
    exit /b 1
)
if not exist "%PDF_PARSER_PYTHON%" (
    echo Failed to find PDF parser Python interpreter: %PDF_PARSER_PYTHON%
    exit /b 1
)
if not exist "%PROJECT_ROOT%\app\main.py" (
    echo Missing backend entry: %PROJECT_ROOT%\app\main.py
    exit /b 1
)
"%NLP_RAG_PYTHON%" -c "import uvicorn, app.main" >nul 2>nul
if errorlevel 1 (
    echo Backend dependency check failed. Verify the nlp-rag environment.
    exit /b 1
)
exit /b 0
