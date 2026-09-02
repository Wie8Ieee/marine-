$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
$kernelSlug = "rawanabuakleh1/marine-ssdlite-seed-42-sequence-safe"
$outputDir = Join-Path $projectRoot "training_results\ssdlite_seed_42_kaggle"
$logPath = Join-Path $projectRoot "training_results\ssdlite_seed_42_kaggle_watcher.log"

New-Item -ItemType Directory -Force (Split-Path -Parent $logPath) | Out-Null

function Write-WatcherLog([string]$message) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -LiteralPath $logPath -Value "[$timestamp] $message"
}

Write-WatcherLog "Watcher started for $kernelSlug"

while ($true) {
    try {
        $statusText = (& $pythonExe -m kaggle kernels status $kernelSlug 2>&1 | Out-String).Trim()
        Write-WatcherLog $statusText

        if ($statusText -match "COMPLETE") {
            New-Item -ItemType Directory -Force $outputDir | Out-Null
            & $pythonExe -m kaggle kernels output $kernelSlug -p $outputDir 2>&1 |
                ForEach-Object { Write-WatcherLog $_ }
            Write-WatcherLog "Download completed: $outputDir"
            exit 0
        }

        if ($statusText -match "ERROR|CANCEL|FAILED") {
            Write-WatcherLog "Kernel ended without successful completion; outputs were not downloaded."
            exit 1
        }
    }
    catch {
        Write-WatcherLog "Transient watcher error: $($_.Exception.Message)"
    }

    Start-Sleep -Seconds 60
}
