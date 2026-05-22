[CmdletBinding()]
param(
    [int]$Port
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

Write-Host "Gateway port: $Port"

$connections = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
if (-not $connections) {
    Write-Host "Status: port is free"
    exit 0
}

$processIds = $connections | Select-Object -ExpandProperty OwningProcess -Unique
Write-Host "Status: port is occupied"

foreach ($processId in $processIds) {
    try {
        $process = Get-Process -Id $processId -ErrorAction Stop
        Write-Host ("Process: PID={0} Name={1}" -f $process.Id, $process.ProcessName)
    } catch {
        Write-Host ("Process: PID={0} exited" -f $processId)
    }
}
