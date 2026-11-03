@echo off
chcp 65001 >nul
rem ===========================================================================
rem  Polyglot Media Summarizer - one-click local launcher (CMD entry point).
rem
rem  Why this wrapper exists:
rem    Running a .ps1 directly from cmd.exe pops up the "choose an app to open
rem    this .ps1 file" dialog, and double-clicking a .ps1 may hit the execution
rem    policy. This .bat calls the same script with -ExecutionPolicy Bypass,
rem    so it works from cmd.exe and by double-click, without changing the
rem    machine-wide policy.
rem
rem  All logic lives in start-local.ps1. Edit that file, not this one.
rem
rem  Usage (run from anywhere):
rem      scripts\start-local.bat
rem      scripts\start-local.bat -Port 8001
rem      scripts\start-local.bat -Reload -NoBrowser
rem      scripts\start-local.bat -Setup            (first run: builds .venv)
rem      scripts\start-local.bat -BindAddress 0.0.0.0
rem
rem  Press Ctrl+C to stop the server.
rem ===========================================================================
setlocal
set "SCRIPT_DIR=%~dp0"

rem Prefer the absolute path: if PATH is broken, a bare "powershell" may not
rem resolve, and this script would fail before it can report anything useful.
set "PS=powershell"
if exist "%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" set "PS=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"

"%PS%" -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%start-local.ps1" %*
set "RC=%ERRORLEVEL%"

rem 1 = the launcher refused to start (no venv / port busy / bad Python).
rem Keep the window open so the reason stays visible; Ctrl+C during a normal
rem run returns other codes and should just close.
if "%RC%"=="1" pause
exit /b %RC%
