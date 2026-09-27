# Opens the project with its venv active. Launched by shell.bat, but you can
# also dot-source it from an existing shell:  . .\src\shell.ps1
#
# $PSScriptRoot is this file's folder (...\no phones\src), so the parent is
# the project root. Deriving it rather than hardcoding means this keeps
# working if you move or rename the project folder.

$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root

$activate = Join-Path $root ".venv\Scripts\Activate.ps1"

if (-not (Test-Path $activate)) {
    Write-Host ""
    Write-Host "No .venv found at $activate" -ForegroundColor Red
    Write-Host "Rebuild it with:" -ForegroundColor Yellow
    Write-Host "  py -3.12 -m venv .venv" -ForegroundColor Yellow
    Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host "  python -m pip install -r requirements.txt" -ForegroundColor Yellow
    Write-Host ""
    return
}

& $activate

$exe = (python -c "import sys; print(sys.executable)")

Write-Host ""
Write-Host "  no phones" -ForegroundColor Cyan
Write-Host "  python:  $exe" -ForegroundColor DarkGray
Write-Host "  cwd:     $root" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  python src\detect.py                     " -NoNewline -ForegroundColor Green
Write-Host "# branch A + logging" -ForegroundColor DarkGray
Write-Host "  python src\detect.py --model yolo11n.pt  " -NoNewline -ForegroundColor Green
Write-Host "# compare speed" -ForegroundColor DarkGray
Write-Host "  python src\capture.py                    " -NoNewline -ForegroundColor Green
Write-Host "# webcam only" -ForegroundColor DarkGray
Write-Host ""
