@echo off
rem Builds the launchers into ..\release.  Nothing of the scientific stack is
rem packaged: the result is a stdlib-only shim that starts the project's venv.
rem
rem ASCII only and no chcp, for the same reason as fly.cmd: cmd.exe tracks its
rem position inside a batch file in bytes.
setlocal
set "ROOT=%~dp0.."
set "PY=%ROOT%\.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo Virtual environment not found: "%PY%"
  exit /b 1
)
"%PY%" "%~dp0build.py"
