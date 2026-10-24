@echo off
rem Double-click this to get a console that STAYS open, already sitting in the
rem project folder with `fly` available as a command for this session only.
rem
rem ASCII only, no chcp: see the note in fly.cmd.
cd /d "%~dp0"
set "PATH=%~dp0;%PATH%"
call "%~dp0fly.cmd"
cmd /k
