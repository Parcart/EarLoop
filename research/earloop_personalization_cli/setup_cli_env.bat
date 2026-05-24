@echo off
setlocal
cd /d "%~dp0"

echo == Creating minimal CLI dev env ==
if not exist ".venv-cli\Scripts\python.exe" (
  py -3.11 -m venv .venv-cli
  if errorlevel 1 (
    echo Failed to create .venv-cli with py -3.11
    echo Try: python -m venv .venv-cli
    pause
    exit /b 1
  )
)

.venv-cli\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.venv-cli\Scripts\python.exe -m pip install -r requirements.txt

echo.
echo Done. Use this interpreter in PyCharm:
echo %CD%\.venv-cli\Scripts\python.exe
echo.
pause
