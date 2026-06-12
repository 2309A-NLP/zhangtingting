@echo off
setlocal

cd /d "%~dp0"

set "PROJECT_ROOT=%CD%"
set "API_BASE=http://127.0.0.1:8000/api"
set "COLLECTION_NAME=prospectus_chunks_04"

if /I "%~1"=="check" goto check

call :validate || exit /b 1
call :require_backend || exit /b 1

echo Step 1/2: Resetting active collection (%COLLECTION_NAME%)...
curl.exe -fsS -X POST "%API_BASE%/reset-index"
if errorlevel 1 exit /b 1

echo.
echo Step 2/2: Building unified heavy PDF index...
curl.exe -fsS -X POST "%API_BASE%/ingest?force=true"
exit /b %errorlevel%

:check
call :validate || exit /b 1
echo Reingest script check passed.
echo Commands:
echo   curl.exe -fsS -X POST "%API_BASE%/reset-index"
echo   curl.exe -fsS -X POST "%API_BASE%/ingest?force=true"
exit /b 0

:validate
where curl.exe >nul 2>nul
if errorlevel 1 (
    echo Failed to find curl.exe in PATH.
    exit /b 1
)
exit /b 0

:require_backend
curl.exe -fsS "%API_BASE%/../.." >nul 2>nul
if errorlevel 1 (
    echo Backend is not reachable at http://127.0.0.1:8000/
    exit /b 1
)
exit /b 0
