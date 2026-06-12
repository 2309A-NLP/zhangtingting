# NLP-RAG Deployment Verification Test Script

param(
    [switch]$Quick,
    [switch]$Full
)

$ErrorActionPreference = "Continue"

function Write-TestHeader {
    param($msg)
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host " $msg" -ForegroundColor Cyan
    Write-Host "========================================`n" -ForegroundColor Cyan
}

function Test-ContainerRunning {
    param($name)
    $status = docker inspect $name --format='{{.State.Running}}' 2>$null
    return $status -eq "true"
}

function Invoke-HttpCheck {
    param($url, $expectedCode = 200)
    try {
        $response = Invoke-WebRequest -Uri $url -Method Get -TimeoutSec 10 -UseBasicParsing -ErrorAction Stop
        return $response.StatusCode -eq $expectedCode
    }
    catch {
        return $false
    }
}

# Test 1: Docker Environment Check
Write-TestHeader "Test 1: Docker Environment Check"

$dockerVersion = docker --version
Write-Host "Docker version: $dockerVersion"

$composeVersion = docker compose version
Write-Host "Docker Compose version: $composeVersion"

# Test 2: Container Status Check
Write-TestHeader "Test 2: Container Status Check"

$containers = @(
    "nlp-rag-etcd",
    "nlp-rag-minio",
    "nlp-rag-mongodb",
    "nlp-rag-milvus",
    "nlp-rag-backend",
    "nlp-rag-frontend"
)

$allRunning = $true
foreach ($container in $containers) {
    $running = Test-ContainerRunning $container
    $status = if ($running) { "Running" } else { "Not running" }
    $color = if ($running) { "Green" } else { "Red" }
    $icon = if ($running) { "[OK]" } else { "[X]" }
    Write-Host "$icon $container : $status" -ForegroundColor $color
    if (-not $running) { $allRunning = $false }
}

# Test 3: Service Health Check
Write-TestHeader "Test 3: Service Health Check"

$healthChecks = @(
    @{ Name = "Backend API"; Url = "http://localhost:8000/"; Icon = "backend" },
    @{ Name = "Frontend UI"; Url = "http://localhost:7860/"; Icon = "frontend" },
    @{ Name = "Milvus Console"; Url = "http://localhost:9091/healthz"; Icon = "milvus" },
    @{ Name = "MinIO Console"; Url = "http://localhost:9100/minio/health/live"; Icon = "minio" }
)

$allHealthy = $true
foreach ($check in $healthChecks) {
    $healthy = Invoke-HttpCheck -Url $check.Url
    $status = if ($healthy) { "Healthy" } else { "Error" }
    $color = if ($healthy) { "Green" } else { "Red" }
    $icon = if ($healthy) { "[OK]" } else { "[X]" }
    Write-Host "$icon $($check.Name) ($($check.Url)): $status" -ForegroundColor $color
    if (-not $healthy) { $allHealthy = $false }
}

# Test 4: API Function Tests
if (-not $Quick) {
    Write-TestHeader "Test 4: API Function Tests"

    Write-Host "Testing backend root path..."
    $rootOk = Invoke-HttpCheck -Url "http://localhost:8000/"
    Write-Host "  Root path: $(if($rootOk){'OK'}else{'Error'})" -ForegroundColor $(if($rootOk){'Green'}else{'Red'})

    Write-Host "Testing Swagger docs..."
    $swaggerOk = Invoke-HttpCheck -Url "http://localhost:8000/docs"
    Write-Host "  Swagger UI: $(if($swaggerOk){'OK'}else{'Error'})" -ForegroundColor $(if($swaggerOk){'Green'}else{'Red'})

    Write-Host "Testing OpenAPI spec..."
    $openapiOk = Invoke-HttpCheck -Url "http://localhost:8000/openapi.json"
    Write-Host "  OpenAPI JSON: $(if($openapiOk){'OK'}else{'Error'})" -ForegroundColor $(if($openapiOk){'Green'}else{'Red'})
}

# Test 5: Data Persistence Check
if (-not $Quick) {
    Write-TestHeader "Test 5: Data Persistence Check"

    $volumes = @("etcd", "minio", "mongodb", "milvus")
    foreach ($vol in $volumes) {
        $path = ".\volumes\$vol"
        $exists = Test-Path $path
        $status = if ($exists) { "Created" } else { "Not created" }
        $color = if ($exists) { "Green" } else { "Yellow" }
        Write-Host "$vol volume: $status" -ForegroundColor $color
    }
}

# Test 6: Network Connectivity
if (-not $Quick) {
    Write-TestHeader "Test 6: Container Network Connectivity"

    Write-Host "Testing backend to MongoDB connection..."
    $result = docker compose exec -T backend ping -c 1 mongodb 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  backend -> MongoDB: OK" -ForegroundColor Green
    } else {
        Write-Host "  backend -> MongoDB: Error" -ForegroundColor Red
    }

    Write-Host "Testing backend to Milvus connection..."
    $result = docker compose exec -T backend ping -c 1 milvus 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  backend -> Milvus: OK" -ForegroundColor Green
    } else {
        Write-Host "  backend -> Milvus: Error" -ForegroundColor Red
    }
}

# Summary Report
Write-TestHeader "Deployment Verification Summary"

if ($allRunning -and $allHealthy) {
    Write-Host "[OK] All tests passed! Deployment successful!" -ForegroundColor Green
    Write-Host "`nYou can access services at:" -ForegroundColor Cyan
    Write-Host "  - Frontend UI: http://localhost:7860"
    Write-Host "  - Backend API: http://localhost:8000"
    Write-Host "  - Swagger Docs: http://localhost:8000/docs"
    Write-Host "  - Milvus Console: http://localhost:9091"
    exit 0
}
else {
    Write-Host "[X] Some tests failed, please check logs" -ForegroundColor Red
    Write-Host "`nView logs command:" -ForegroundColor Yellow
    Write-Host "  docker compose logs backend"
    Write-Host "  docker compose logs frontend"
    exit 1
}
