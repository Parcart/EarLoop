@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-cli\Scripts\python.exe" (
  call setup_cli_env.bat
)
.venv-cli\Scripts\python.exe main.py devices
pause
