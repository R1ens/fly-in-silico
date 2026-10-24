@echo off
rem Adds D:\fly-brain to the *user* PATH so `fly` works from any directory.
rem Touches only this user's environment. Open a new console afterwards.
rem ASCII only, no chcp: see the note in fly.cmd.
setlocal
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

for /f "skip=2 tokens=2,*" %%a in ('reg query HKCU\Environment /v PATH 2^>nul') do set "USERPATH=%%b"

echo %USERPATH% | find /i "%ROOT%" >nul
if not errorlevel 1 (
  echo "%ROOT%" is already in the user PATH. Nothing changed.
  exit /b 0
)

if defined USERPATH (
  setx PATH "%USERPATH%;%ROOT%" >nul
) else (
  setx PATH "%ROOT%" >nul
)
if errorlevel 1 (
  echo Could not change PATH.
  exit /b 1
)

echo Added to the user PATH: %ROOT%
echo Open a new console, then this works anywhere:  fly doctor
endlocal
