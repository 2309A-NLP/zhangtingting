# NLP-RAG Health Check Script

param(
    [switch]$Watch,
    [switch]$Detail
)

$ErrorActionPreference = "Continue"

function Test-ServiceHealth {
    param($Name, $Url, $ExpectedStatus = 200)

    try {
        $response = Invoke-WebRequest -Uri $Url -Method Get -TimeoutSec 10 -UseBasicParsing -ErrorAction Stop
        $status = if ($response.StatusCode -eq $ExpectedStatus) { "Healthy" } else { "Error" }
        return @{ Name = $Name; Url = $Url; Status = $status; Code = $response.StatusCode; Message = "OK" }
    }
    catch {
        return @{ Name = $Name; Url = $Url; Status = "Error"; Code = 0; Message = $_.Exception.Message }
    }
}

function Get-ServiceStatus {
    $services = @(
        @{ Name = "Backend API"; Url = "http://localhost:8000/" },
        @{ Name = "Frontend UI"; Url = "http://localhost:7860/" },
        @{ Name = "Milvus Health"; Url = "http://localhost:9091/healthz" },
        @{ Name = "MinIO Health"; Url = "http://localhost:9100/minio/health/live" }
    )

    Write-Host "`n========== Service Health Check ==========`n" -ForegroundColor Cyan

    $results = @()
    foreach ($svc in $services) {
        $result = Test-ServiceHealth -Name $svc.Name -Url $svc.Url
        $results += $result

        $color = switch ($result.Status) {
            "Healthy" { "Green" }
            "Error" { "Red" }
            default { "Yellow" }
        }

        $icon = switch ($result.Status) {
            "Healthy" { "[OK]" }
            "Error" { "[X]" }
            default { "[?]" }
        }

        Write-Host "$icon $($result.Name)" -ForegroundColor $color
        Write-Host "  URL: $($result.Url)"
        Write-Host "  Status: $($result.Status) (HTTP $($result.Code))"
        if ($result.Message -ne "OK") {
            Write-Host "  Error: $($result.Message)" -ForegroundColor Yellow
        }
        Write-Host ""
    }

    Write-Host "========== Container Status ==========`n" -ForegroundColor Cyan
    docker compose ps | Format-Table -AutoSize

    $healthy = ($results | Where-Object { $_.Status -eq "Healthy" }).Count
    $total = $results.Count

    Write-Host "========== Summary ==========`n" -ForegroundColor Cyan
    Write-Host "Healthy services: $healthy / $total"

    if ($healthy -eq $total) {
        Write-Host "`nAll services are running normally!" -ForegroundColor Green
        return 0
    }
    else {
        Write-Host "`nSome services have issues, please check logs" -ForegroundColor Yellow
        return 1
    }
}

function Watch-Services {
    Write-Host "Continuous monitoring mode (Ctrl+C to exit)`n" -ForegroundColor Yellow

    while ($true) {
        Clear-Host
        $exitCode = Get-ServiceStatus
        Write-Host "Next check: in 30 seconds..." -ForegroundColor DarkGray
        Start-Sleep -Seconds 30
    }
}

if ($Watch) {
    Watch-Services
}
else {
    $exitCode = Get-ServiceStatus
    exit $exitCode
}
