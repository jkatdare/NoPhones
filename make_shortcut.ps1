# Creates a "No Phones" icon on your desktop that starts the monitor.
# Run once:  powershell -ExecutionPolicy Bypass -File make_shortcut.ps1

$root = $PSScriptRoot
$desktop = [Environment]::GetFolderPath("Desktop")
$path = Join-Path $desktop "No Phones.lnk"

$shell = New-Object -ComObject WScript.Shell
$link = $shell.CreateShortcut($path)
$link.TargetPath = Join-Path $root "No Phones.bat"
$link.WorkingDirectory = $root
$link.IconLocation = (Join-Path $root "assets\no_phones.ico") + ",0"
$link.Description = "Alerts you when you pick up your phone"
$link.Save()

Write-Host "Created $path"
