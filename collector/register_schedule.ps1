# Registers all automatic data-capture tasks in Windows Task Scheduler.
# Run:  powershell -ExecutionPolicy Bypass -File collector\register_schedule.ps1
# Remove all:  Get-ScheduledTask -TaskName 'ResearchFunds *' | Unregister-ScheduledTask -Confirm:$false
# Times are LOCAL. Designed for US Central; adjust if your machine is elsewhere.

$repo = Split-Path -Parent $PSScriptRoot
$settings = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 2)

$jobs = @(
    @{ Name = 'ResearchFunds Options Snapshot W1'; Bat = 'run_snapshot.bat'; Args = '--window W1'; At = '08:15' },
    @{ Name = 'ResearchFunds Options Snapshot W2'; Bat = 'run_snapshot.bat'; Args = '--window W2'; At = '10:30' },
    @{ Name = 'ResearchFunds Options Snapshot W3'; Bat = 'run_snapshot.bat'; Args = '--window W3'; At = '14:45' },
    @{ Name = 'ResearchFunds Options Snapshot W4'; Bat = 'run_snapshot.bat'; Args = '--window W4'; At = '15:10' },
    @{ Name = 'ResearchFunds Options Trades';      Bat = 'run_snapshot_trades.bat'; Args = '';    At = '15:25' },
    @{ Name = 'ResearchFunds Nightly Refresh';     Bat = 'run_refresh.bat'; Args = '';           At = '17:30' }
)

foreach ($j in $jobs) {
    $action = New-ScheduledTaskAction -Execute (Join-Path $PSScriptRoot $j.Bat) `
        -Argument $j.Args -WorkingDirectory $repo
    $trigger = New-ScheduledTaskTrigger -Weekly `
        -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At $j.At
    Register-ScheduledTask -TaskName $j.Name -Action $action -Trigger $trigger `
        -Settings $settings -Force | Out-Null
    Write-Output "registered: $($j.Name) at $($j.At)"
}
Write-Output "Done. The machine must be on or asleep (not shut down) at capture times."
