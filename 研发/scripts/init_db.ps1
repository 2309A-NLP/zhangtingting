$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$envFile = Join-Path $root ".env"

if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if (-not $_ -or $_.Trim().StartsWith("#")) {
            return
        }
        $parts = $_ -split "=", 2
        if ($parts.Length -eq 2) {
            [System.Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1].Trim(), "Process")
        }
    }
}

$dbName = if ($env:MYSQL_DATABASE) { $env:MYSQL_DATABASE } else { "rag_app" }
$rootPassword = if ($env:MYSQL_ROOT_PASSWORD) { $env:MYSQL_ROOT_PASSWORD } else { "root_password" }

Write-Host "Applying schema to database: $dbName"
Get-Content (Join-Path $PSScriptRoot "mysql-init\01_schema.sql") | docker compose exec -T mysql mysql -uroot "-p$rootPassword" $dbName
Get-Content (Join-Path $PSScriptRoot "mysql-init\02_seed_roles.sql") | docker compose exec -T mysql mysql -uroot "-p$rootPassword" $dbName
Write-Host "Database initialization completed."
