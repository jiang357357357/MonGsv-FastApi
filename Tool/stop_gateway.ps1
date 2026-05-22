[CmdletBinding()]
param(
    [int]$Port,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Get-RepoRoot {
    return (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
}

function Get-MonConfigPath {
    return Join-Path (Get-RepoRoot) ".monconfig"
}

function Get-MonConfigValue {
    param(
        [string]$Section,
        [string]$Key,
        [string]$Default = ""
    )

    $configPath = Get-MonConfigPath
    if (-not (Test-Path $configPath)) {
        return $Default
    }

    $currentSection = ""
    foreach ($rawLine in Get-Content -LiteralPath $configPath -Encoding UTF8) {
        $line = $rawLine.Trim()
        if (-not $line -or $line.StartsWith("#")) {
            continue
        }
        if ($line.StartsWith("[") -and $line.EndsWith("]")) {
            $currentSection = $line.Substring(1, $line.Length - 2).Trim()
            continue
        }
        if ($currentSection -ne $Section) {
            continue
        }
        $commentIndex = $line.IndexOf("#")
        if ($commentIndex -ge 0) {
            $line = $line.Substring(0, $commentIndex).Trim()
        }
        if (-not $line.Contains("=")) {
            continue
        }
        $parts = $line.Split("=", 2)
        if ($parts[0].Trim() -eq $Key) {
            return $parts[1].Trim()
        }
    }

    return $Default
}

if (-not $Port) {
    $Port = [int](Get-MonConfigValue -Section "server" -Key "PORT" -Default "40032")
}

$connections = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
if (-not $connections) {
    Write-Host "Port $Port is free. Nothing to stop."
    exit 0
}

$processIds = $connections | Select-Object -ExpandProperty OwningProcess -Unique
Write-Host ("Stopping processes on port {0}: {1}" -f $Port, (($processIds -join ", ")))

if (-not $Force) {
    $answer = Read-Host "Confirm stop? Type Y to continue"
    if ($answer -ne "Y" -and $answer -ne "y") {
        Write-Host "Cancelled"
        exit 1
    }
}

foreach ($processId in $processIds) {
    try {
        Stop-Process -Id $processId -Force
        Write-Host ("Stopped PID={0}" -f $processId)
    } catch {
        Write-Host ("Failed to stop PID={0}: {1}" -f $processId, $_.Exception.Message)
    }
}
