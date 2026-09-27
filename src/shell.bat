@echo off
REM Double-click this to open a PowerShell sitting in the project root with
REM the venv already active.
REM
REM All the real work is in shell.ps1 next to this file. Keeping the batch
REM side to one line avoids the quoting traps you hit when continuing a
REM quoted -Command across multiple lines with ^.
REM
REM -NoExit  keeps the window open and interactive after the script runs
REM -ExecutionPolicy Bypass  applies to this process only, nothing system-wide

powershell.exe -NoExit -ExecutionPolicy Bypass -File "%~dp0shell.ps1"
