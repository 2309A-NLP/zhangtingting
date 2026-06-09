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
if /I "%~1"=="check" goto check

call :validate || exit /b 1

echo Starting frontend with %NLP_RAG_PYTHON% ...
"%NLP_RAG_PYTHON%" frontend\gradio_app.py
exit /b %errorlevel%

:check
call :validate || exit /b 1
echo Frontend script check passed.
echo Python: %NLP_RAG_PYTHON%
echo Command: "%NLP_RAG_PYTHON%" frontend\gradio_app.py
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
if not exist "%PROJECT_ROOT%\frontend\gradio_app.py" (
    echo Missing frontend entry: %PROJECT_ROOT%\frontend\gradio_app.py
    exit /b 1
)
"%NLP_RAG_PYTHON%" -c "import gradio, requests" >nul 2>nul
if errorlevel 1 (
    echo Frontend dependency check failed in interpreter: %NLP_RAG_PYTHON%
    exit /b 1
)
exit /b 0
