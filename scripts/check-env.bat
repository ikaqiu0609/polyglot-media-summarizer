@echo off
chcp 65001 >nul
rem ===========================================================================
rem  Windows environment check - CMD entry point (double-click friendly).
rem
rem  Why this wrapper exists:
rem    Running a .ps1 directly from cmd.exe pops up the "choose an app to open
rem    this .ps1 file" dialog, and double-clicking a .ps1 may hit the execution
rem    policy. This .bat calls the same script with -ExecutionPolicy Bypass,
rem    so it works from cmd.exe and by double-click, without changing the
rem    machine-wide policy.
rem
rem  All logic lives in check-env.ps1 (8 checks). Edit that file, not this one.
rem
rem  Usage (run from anywhere):
rem      scripts\check-env.bat
rem      scripts\check-env.bat -Python "D:\Python312\python.exe"
rem
rem  Exit code: same as check-env.ps1 (1 = at least one [FAIL]).
rem ===========================================================================
setlocal
set "SCRIPT_DIR=%~dp0"

rem Prefer the absolute path: if PATH itself is broken (the very thing we are
rem checking), a bare "powershell" would not resolve.
set "PS=powershell"
if exist "%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" set "PS=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"

"%PS%" -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%check-env.ps1" %*
set "RC=%ERRORLEVEL%"

echo.
rem Keep the window open when launched by double-click / from Explorer.
pause
exit /b %RC%
