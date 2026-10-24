@echo off
rem Entry point: runs the CLI from the project's own virtual environment.
rem
rem This file is deliberately ASCII-only and never calls chcp.  cmd.exe tracks
rem its position inside a batch file in bytes, so changing the code page while
rem the file is being read splits the remaining lines in the wrong places and
rem executes the pieces as commands.  Python writes its Russian output using
rem the console's own code page, which needs no help from here.
setlocal
set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo Virtual environment not found: "%PY%"
  echo.
  echo Build it:
  echo   uv venv --python 3.12 "%ROOT%.venv"
  echo   uv pip install --python "%PY%" --index-url https://download.pytorch.org/whl/cu128 torch
  echo   uv pip install --python "%PY%" "pandas^>=2.2,^<3" "numpy^<3" pyarrow scipy tqdm
  exit /b 1
)

pushd "%ROOT%lab"
"%PY%" -m flybrain.cli %*
set "ERR=%ERRORLEVEL%"
popd

if not "%ERR%"=="0" (
  "%PY%" -V >nul 2>&1
  if errorlevel 1 (
    echo.
    echo The venv interpreter does not start at all.
    echo Run "%ROOT%fix-venv.cmd" to repair it.
  )
)

rem Double-clicked from Explorer: the console closes the moment this exits, so
rem hold it open. Detected by this file's name appearing in the command line
rem that started cmd.exe, together with no arguments of our own.
if "%~1"=="" (
  echo %cmdcmdline% | "%SystemRoot%\System32\find.exe" /i "%~nx0" >nul
  if not errorlevel 1 (
    echo.
    echo Tip: double-click console.cmd for a window that stays open.
    pause
  )
)

exit /b %ERR%
