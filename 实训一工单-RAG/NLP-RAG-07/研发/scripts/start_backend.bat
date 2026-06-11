@echo off
setlocal

cd /d "%~dp0"
cd ..
set "PROJECT_ROOT=%CD%"

if not "%NLP_RAG_PYTHON%"=="" goto python_ready
if not "%CONDA_PREFIX%"=="" if exist "%CONDA_PREFIX%\python.exe" set "NLP_RAG_PYTHON=%CONDA_PREFIX%\python.exe"
if not "%NLP_RAG_ENV_NAME%"=="" goto resolve_by_name
set "NLP_RAG_ENV_NAME=nlp-rag"

:resolve_by_name
if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" call "%USERPROFILE%\anaconda3\condabin\conda.bat" activate "%NLP_RAG_ENV_NAME%" >nul 2>nul
if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" call "%USERPROFILE%\miniconda3\condabin\conda.bat" activate "%NLP_RAG_ENV_NAME%" >nul 2>nul
if exist "D:\anaconda2024\condabin\conda.bat" call "D:\anaconda2024\condabin\conda.bat" activate "%NLP_RAG_ENV_NAME%" >nul 2>nul
if not "%CONDA_PREFIX%"=="" if exist "%CONDA_PREFIX%\python.exe" set "NLP_RAG_PYTHON=%CONDA_PREFIX%\python.exe"
if not "%NLP_RAG_PYTHON%"=="" goto python_ready
if exist "D:\anaconda2024\envs\%NLP_RAG_ENV_NAME%\python.exe" set "NLP_RAG_PYTHON=D:\anaconda2024\envs\%NLP_RAG_ENV_NAME%\python.exe"
if exist "%USERPROFILE%\anaconda3\envs\%NLP_RAG_ENV_NAME%\python.exe" set "NLP_RAG_PYTHON=%USERPROFILE%\anaconda3\envs\%NLP_RAG_ENV_NAME%\python.exe"
if exist "%USERPROFILE%\miniconda3\envs\%NLP_RAG_ENV_NAME%\python.exe" set "NLP_RAG_PYTHON=%USERPROFILE%\miniconda3\envs\%NLP_RAG_ENV_NAME%\python.exe"

:python_ready
if not "%PDF_PARSER_PYTHON%"=="" goto parser_ready
set "PDF_PARSER_PYTHON=D:\anaconda2024\envs\pdf-parser\python.exe"

:parser_ready
if not "%PDF_PARSER_BACKEND%"=="" goto store_ready
set "PDF_PARSER_BACKEND=parse2"

:store_ready
if not "%CONVERSATION_STORE_BACKEND%"=="" goto redis_ready
set "CONVERSATION_STORE_BACKEND=redis"

:redis_ready
if not "%REDIS_URI%"=="" goto backend_ready
set "REDIS_URI=redis://127.0.0.1:6379/0"

:backend_ready
if /I "%~1"=="check" goto check

call :validate || exit /b 1

for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo Stopping existing process on port 8000: PID %%p
    taskkill /PID %%p /F >nul 2>nul
)

echo Starting backend on http://127.0.0.1:8000 ...
echo Python: %NLP_RAG_PYTHON%
echo Conversation store: %CONVERSATION_STORE_BACKEND%
echo Redis URI: %REDIS_URI%
echo Query mode: unified default corpus routing
"%NLP_RAG_PYTHON%" -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
exit /b %errorlevel%

:check
call :validate || exit /b 1
echo Backend script check passed.
echo Python: %NLP_RAG_PYTHON%
echo Conversation store: %CONVERSATION_STORE_BACKEND%
echo Redis URI: %REDIS_URI%
echo Query mode: unified default corpus routing
echo Command: "%NLP_RAG_PYTHON%" -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
exit /b 0

:validate
if "%NLP_RAG_PYTHON%"=="" (
    echo Failed to resolve Python interpreter for env: %NLP_RAG_ENV_NAME%
    exit /b 1
)
if not exist "%NLP_RAG_PYTHON%" (
    echo Failed to find Python interpreter: %NLP_RAG_PYTHON%
    exit /b 1
)
if not exist "%PDF_PARSER_PYTHON%" (
    echo Failed to find PDF parser Python interpreter: %PDF_PARSER_PYTHON%
    exit /b 1
)
if not exist "%PROJECT_ROOT%\backend\main.py" (
    echo Missing backend entry: %PROJECT_ROOT%\backend\main.py
    exit /b 1
)
if /I "%CONVERSATION_STORE_BACKEND%"=="redis" (
    "%NLP_RAG_PYTHON%" -c "import redis; redis.Redis.from_url(r'%REDIS_URI%').ping()" >nul 2>nul
    if errorlevel 1 (
        echo Redis check failed. Start Redis first or set CONVERSATION_STORE_BACKEND=memory
        exit /b 1
    )
)
"%NLP_RAG_PYTHON%" -c "import uvicorn, backend.main" >nul 2>nul
if errorlevel 1 (
    echo Backend dependency check failed in interpreter: %NLP_RAG_PYTHON%
    exit /b 1
)
exit /b 0
