@echo off
rem Repairs the venv when every command dies with
rem   uv trampoline failed to spawn Python child process
rem
rem uv keeps the path to the base interpreter inside the trampoline
rem .venv\Scripts\python.exe, and that path points at a directory without the
rem patch number (cpython-3.12-windows-x86_64-none), which is a junction to the
rem real one (cpython-3.12.13-...).  When uv cleans its cache the junction goes
rem away; the trampoline does not read pyvenv.cfg, so editing it does not help.
rem This recreates the junction.
rem
rem ASCII only, and no chcp: see the note in fly.cmd.
setlocal EnableDelayedExpansion
set "ROOT=%~dp0"
set "CFG=%ROOT%.venv\pyvenv.cfg"

if not exist "%CFG%" (
  echo No "%CFG%" - the environment was never built.
  exit /b 1
)

set "HOMEDIR="
for /f "tokens=1,* delims== " %%a in ('type "%CFG%"') do (
  if /i "%%a"=="home" set "HOMEDIR=%%b"
)
if not defined HOMEDIR (
  echo pyvenv.cfg has no "home" line.
  exit /b 1
)
for %%i in ("!HOMEDIR!") do set "BASE=%%~dpi"
for %%i in ("!HOMEDIR!") do set "LEAF=%%~nxi"

rem cpython-3.12.13-windows-x86_64-none -> cpython 3.12.13 windows x86_64 none
for /f "tokens=1-5 delims=-" %%a in ("!LEAF!") do (
  set "IMPL=%%a" & set "VER=%%b" & set "OS=%%c" & set "ARCH=%%d" & set "ABI=%%e"
)
for /f "tokens=1,2 delims=." %%a in ("!VER!") do set "SHORT=%%a.%%b"
set "ALIAS=!BASE!!IMPL!-!SHORT!-!OS!-!ARCH!-!ABI!"

echo real interpreter: !HOMEDIR!
echo alias uv expects: !ALIAS!
echo.

if not exist "!HOMEDIR!\python.exe" (
  echo The interpreter itself is gone. Rebuild the environment:
  echo   uv venv --python 3.12 "%ROOT%.venv"
  exit /b 1
)

if exist "!ALIAS!\python.exe" (
  echo Alias is in place, nothing to repair.
) else (
  if exist "!ALIAS!" rmdir "!ALIAS!" 2>nul
  mklink /J "!ALIAS!" "!HOMEDIR!" >nul
  if errorlevel 1 (
    echo Could not create the junction.
    exit /b 1
  )
  echo Alias recreated.
)

echo.
"%ROOT%.venv\Scripts\python.exe" -V
if errorlevel 1 (
  echo The interpreter still does not start - rebuild the environment.
  exit /b 1
)
echo.
echo Done. Check with:  "%ROOT%fly.cmd" doctor
endlocal
