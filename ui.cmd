@echo off
rem Double-click this: opens the control window in your browser and keeps the
rem connectome loaded in memory, so every run starts instantly.
rem Close this console (or Ctrl+C) to shut the server down.
rem ASCII only, no chcp: see the note in fly.cmd.
cd /d "%~dp0"
call "%~dp0fly.cmd" ui
echo.
echo Server stopped.
pause
