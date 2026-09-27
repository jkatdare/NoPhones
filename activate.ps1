# Convenience wrapper. Dot-source it so the PATH change sticks in your shell:
#
#     . .\activate.ps1
#
# The leading dot matters. Running .\activate.ps1 without it executes the
# script in a child scope; environment variables happen to survive that, but
# the `deactivate` helper and the prompt prefix may not.

$venv = Join-Path $PSScriptRoot ".venv\Scripts\Activate.ps1"

if (-not (Test-Path $venv)) {
    Write-Host "No .venv found. Create it with:" -ForegroundColor Yellow
    Write-Host "  py -3.12 -m venv .venv" -ForegroundColor Yellow
    Write-Host "  python -m pip install -r requirements.txt" -ForegroundColor Yellow
    return
}

& $venv
Write-Host "venv active: $(python -c 'import sys; print(sys.executable)')" -ForegroundColor Green
