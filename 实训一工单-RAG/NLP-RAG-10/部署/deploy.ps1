# NLP-RAG Deployment Script - Windows PowerShell

param(
    [switch]$Build,
    [switch]$NoBuild,
    [switch]$Stop,
    [switch]$Restart,
    [switch]$Logs,
    [switch]$Status,
    [switch]$Clean,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

function Write-Success { param($msg) Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Info { param($msg) Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-Warn { param($msg) Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Err { param($msg) Write-Host "[ERROR] $msg" -ForegroundColor Red }

function Show-Help {
    Write-Host @"
NLP-RAG Finance Q&A System - Deployment Script

Usage: .\deploy.ps1 [options]

Options:
  -Build      Build and start all services (default)
  -NoBuild    Start services only (skip build)
  -Stop       Stop all services
  -Restart    Restart all services
  -Logs       View service logs
  -Status     View service status
  -Clean      Clean all containers and volumes
  -Help       Show this help message

Examples:
  .\deploy.ps1 -Build       # Full build and start
  .\deploy.ps1 -NoBuild     # Quick start (skip build)
  .\deploy.ps1 -Logs        # View logs
  .\deploy.ps1 -Status      # View status

Service Ports:
  - Backend API:    http://localhost:8000
  - Frontend UI:     http://localhost:7860
  - Milvus:         http://localhost:9091
  - Attu:           http://localhost:3011
  - MongoDB:        localhost:27017
  - Mongo Express:  http://localhost:8082
  - MinIO Console:  http://localhost:9101

"@
}

function Test-DockerRunning {
    try {
        docker version | Out-Null
        return $true
    }
    catch {
        Write-Err "Docker is not running or not installed"
        return $false
    }
}

function Test-NvidiaSupport {
    $nvidia = docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Success "NVIDIA GPU support detected"
        return $true
    }
    Write-Warn "No NVIDIA GPU detected, will skip vLLM service"
    return $false
}

function Initialize-Directories {
    Write-Info "Creating necessary directories..."
    $dirs = @("volumes/etcd", "volumes/minio", "volumes/mongodb", "volumes/milvus", "data", "model", "artifacts", "reports", "config")
    foreach ($dir in $dirs) {
        $path = Join-Path $PSScriptRoot $dir
        if (-not (Test-Path $path)) {
            New-Item -ItemType Directory -Path $path -Force | Out-Null
            Write-Info "  Created: $dir"
        }
    }
}

function Build-Images {
    Write-Info "Building Docker images..."
    docker compose build --no-cache backend frontend
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Image build failed"
        exit 1
    }
    Write-Success "Image build completed"
}

function Wait-ForMilvus {
    Write-Info "Waiting for Milvus to be ready..."
    $maxWait = 120
    $waited = 0
    while ($waited -lt $maxWait) {
        $health = docker inspect nlp-rag-milvus --format='{{.State.Health.Status}}' 2>$null
        if ($health -eq "healthy") {
            Write-Success "Milvus is ready"
            return $true
        }
        Start-Sleep -Seconds 5
        $waited += 5
        Write-Info "    Waiting... ($waited/$maxWait sec)"
    }
    Write-Warn "Milvus may not be fully ready, continuing anyway"
    return $false
}

function Start-Services {
    Write-Info "Starting services..."

    Write-Info "  Starting infrastructure services..."
    docker compose up -d etcd minio mongodb
    Start-Sleep -Seconds 5

    Write-Info "  Starting Milvus..."
    docker compose up -d milvus
    Wait-ForMilvus

    if (Test-NvidiaSupport) {
        Write-Info "  Starting vLLM service..."
        docker compose up -d vllm
    }
    else {
        Write-Warn "  Skipping vLLM (requires NVIDIA GPU)"
    }

    Write-Info "  Starting application services..."
    docker compose up -d backend frontend

    Write-Success "All services started"
}

function Stop-Services {
    Write-Info "Stopping services..."
    docker compose down
    Write-Success "All services stopped"
}

function Show-Logs {
    docker compose logs -f --tail=100
}

function Show-Status {
    Write-Host "`n========== Service Status ==========`n" -ForegroundColor Cyan
    docker compose ps

    Write-Host "`n========== Health Check ==========`n" -ForegroundColor Cyan

    $services = @("backend", "frontend", "milvus", "mongodb", "minio")
    foreach ($svc in $services) {
        $containerName = "nlp-rag-$svc"
        $health = docker inspect $containerName --format='{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' 2>$null
        $status = switch ($health) {
            "healthy" { "[OK] Healthy" }
            "unhealthy" { "[X] Unhealthy" }
            "starting" { "[~] Starting" }
            "running" { "[*] Running" }
            default { "[ ] Not running" }
        }
        $color = switch ($health) {
            "healthy" { "Green" }
            "unhealthy" { "Red" }
            "starting" { "Yellow" }
            "running" { "Cyan" }
            default { "Gray" }
        }
        Write-Host "$containerName : $status" -ForegroundColor $color
    }

    Write-Host "`n========== Port Mappings ==========`n" -ForegroundColor Cyan
    Write-Host "Backend API:       http://localhost:8000"
    Write-Host "Frontend UI:      http://localhost:7860"
    Write-Host "Milvus Console:   http://localhost:9091"
    Write-Host "Attu:             http://localhost:3011"
    Write-Host "Mongo Express:    http://localhost:8082"
    Write-Host "MinIO Console:    http://localhost:9101"
}

function Clean-All {
    Write-Warn "This will delete all containers and data volumes..."
    $confirm = Read-Host "Confirm deletion? (y/N)"
    if ($confirm -ne "y") {
        Write-Info "Cancelled"
        return
    }

    Write-Info "Cleaning..."
    docker compose down -v --remove-orphans
    docker system prune -f
    Write-Success "Cleanup completed"
}

# Main flow
if (-not (Test-DockerRunning)) {
    exit 1
}

Initialize-Directories

switch ($true) {
    $Help { Show-Help }
    $Stop { Stop-Services }
    $Status { Show-Status }
    $Logs { Show-Logs }
    $Clean { Clean-All }
    $Restart {
        Stop-Services
        Start-Services
    }
    $Build {
        Build-Images
        Start-Services
    }
    $NoBuild { Start-Services }
    default {
        Build-Images
        Start-Services
        Show-Status
    }
}
