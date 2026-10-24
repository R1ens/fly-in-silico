@echo off
rem Opens the native window.  Double-clicking this file is the normal way in.
rem
rem ASCII only and no chcp, for the same reason as fly.cmd: cmd.exe tracks its
rem position inside a batch file in bytes, so switching the code page while the
rem file is being read splits the remaining lines in the wrong places.
setlocal
set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"
set "PYW=%ROOT%.venv\Scripts\pythonw.exe"

if not exist "%PY%" (
  echo Virtual environment not found: "%PY%"
  echo.
  echo Repair it with fix-venv.cmd, or rebuild:
  echo   uv venv --python 3.12 "%ROOT%.venv"
  pause
  exit /b 1
)

pushd "%ROOT%lab"
if "%~1"=="--console" (
  rem Keeps the console so worker messages and tracebacks stay visible.
  "%PY%" -m flybrain.cli app
  set "ERR=%ERRORLEVEL%"
  popd
  if not "%ERR%"=="0" pause
  exit /b %ERR%
)

rem No console: the window is the whole interface.
start "" "%PYW%" -m flybrain.cli app
popd
