$ErrorActionPreference = "Stop"
$taskName = "PBL Daily Collect"
$projectDir = "C:\Users\siddh\Desktop\PBL"
$bat = Join-Path $projectDir "run_collect.bat"
if (-not (Test-Path $bat)) { throw "Not found: $bat - save run_collect.bat in the project folder first." }

$action  = New-ScheduledTaskAction -Execute $bat -WorkingDirectory $projectDir
$trigger = New-ScheduledTaskTrigger -Daily -At "8:00PM"
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RunOnlyIfNetworkAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 2)

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Description "Collect live mandi prices once per day; runs as soon as possible if a scheduled run was missed" -Force | Out-Null

$s = (Get-ScheduledTask -TaskName $taskName).Settings
Write-Host "Registered '$taskName'."
Write-Host "StartWhenAvailable (catch-up if missed) = $($s.StartWhenAvailable)"
Write-Host "RunOnlyIfNetworkAvailable              = $($s.RunOnlyIfNetworkAvailable)"