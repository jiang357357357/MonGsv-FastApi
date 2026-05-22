[CmdletBinding()]
param(
    [int]$Port,
    [switch]$KeepTemp,
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

function Resolve-MonPath {
    param(
        [string]$Section,
        [string]$Key,
        [string]$Default = ""
    )

    $rawPath = Get-MonConfigValue -Section $Section -Key $Key -Default $Default
    if (-not $rawPath) {
        return ""
    }
    if ([System.IO.Path]::IsPathRooted($rawPath)) {
        return $rawPath
    }
    return (Join-Path (Get-RepoRoot) $rawPath)
}

if (-not $Port) {
    $Port = [int](Get-MonConfigValue -Section "server" -Key "PORT" -Default "40032")
}

$stopScript = Join-Path $PSScriptRoot "stop_gateway.ps1"
& $stopScript -Port $Port -Force:$Force

if ($KeepTemp) {
    Write-Host "Skipped TEMP_DIR cleanup"
    exit 0
}

$tempDir = Resolve-MonPath -Section "paths" -Key "TEMP_DIR" -Default "Data/Temp"
if (-not $tempDir) {
    Write-Host "TEMP_DIR is not configured. Skip cleanup."
    exit 0
}

if (-not (Test-Path $tempDir)) {
    Write-Host "TEMP_DIR does not exist: $tempDir"
    exit 0
}

if (-not $Force) {
    $answer = Read-Host "Confirm TEMP_DIR cleanup? Type Y to continue"
    if ($answer -ne "Y" -and $answer -ne "y") {
        Write-Host "TEMP_DIR cleanup cancelled"
        exit 1
    }
}

Get-ChildItem -LiteralPath $tempDir -Force -ErrorAction SilentlyContinue | ForEach-Object {
    Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction Stop
}

Write-Host "TEMP_DIR cleanup complete: $tempDir"
