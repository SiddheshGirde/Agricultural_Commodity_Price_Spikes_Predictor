# Registers a Windows scheduled task that runs scripts\run_collect.bat daily at 8 PM, and as soon as
# possible after a missed run. Works from any location. Run once:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_task.ps1
$ErrorActionPreference = "Stop"
$taskName = "PBL Daily Collect"
$projectDir = Split-Path -Parent $PSScriptRoot
$bat = Join-Path $PSScriptRoot "run_collect.bat"
if (-not (Test-Path $bat)) { throw "Not found: $bat" }
if (-not (Test-Path (Join-Path $projectDir ".venv\Scripts\python.exe"))) {
    Write-Warning "No .venv found in $projectDir - create it first (see README), the task runs .venv\Scripts\python.exe"
}

$action   = New-ScheduledTaskAction -Execute $bat -WorkingDirectory $projectDir
$trigger  = New-ScheduledTaskTrigger -Daily -At "8:00PM"
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RunOnlyIfNetworkAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 2)

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Description "Collect live mandi prices once per day; runs as soon as possible if a scheduled run was missed" -Force | Out-Null

$s = (Get-ScheduledTask -TaskName $taskName).Settings
Write-Host "Registered '$taskName' for $projectDir"
Write-Host "StartWhenAvailable (catch-up if missed) = $($s.StartWhenAvailable)"
Write-Host "RunOnlyIfNetworkAvailable              = $($s.RunOnlyIfNetworkAvailable)"