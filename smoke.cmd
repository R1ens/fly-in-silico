@echo off
rem Checks that every command still starts. ASCII only, no chcp: see fly.cmd.
setlocal EnableDelayedExpansion
set "FLY=%~dp0fly.cmd"
set FAIL=0

call "%FLY%" >nul 2>&1
if errorlevel 1 (
  echo   [FAIL] no arguments
  set /a FAIL+=1
) else (
  echo   [ ok ] no arguments
)

call :run "doctor" doctor
call :run "regions" regions
call :run "select" select "type:MBON"
call :run "report" report demo-sugar --no-open
call :run "run" run -a "type:LPLC2" -r 150 --t 0.3 --out smoke-run --no-open
call :run "arena" arena --t 1 --no-comment --out smoke-arena --no-open
call :run "arena ALPN" arena --t 1 --motor ALPN --dn-tau 0.3 --no-comment --out smoke-alpn --no-open
call :run "lateral" lateral --t 2 --top 5 --out smoke-lateral
call :run "index" index --no-open

echo.
if "%FAIL%"=="0" (echo ALL CHECKS PASSED) else (echo FAILURES: %FAIL%)
exit /b %FAIL%

rem %1 -- label, the rest goes to fly.cmd
:run
set "NAME=%~1"
set "ARGS=%*"
set "ARGS=!ARGS:*%1 =!"
if "!ARGS!"=="%*" set "ARGS="
call "%FLY%" !ARGS! >nul 2>&1
if errorlevel 1 (
  echo   [FAIL] !NAME!
  set /a FAIL+=1
) else (
  echo   [ ok ] !NAME!
)
exit /b 0
