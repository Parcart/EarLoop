@echo off
cd /d %~dp0
py -3.11 build_cli_package.py --recreate-venv
pause
