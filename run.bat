@echo off
chcp 65001 >nul
rem ===========================================================================
rem  Polyglot Media Summarizer - the only thing you need to run after setup.
rem
rem  First time?  Run the deployment steps in README "B" once. After that,
rem  just double-click this file (or type "run.bat" in cmd).
rem
rem  Name note: deliberately NOT "start.bat" -- "start" is a cmd built-in and
rem  would open a new window instead of running this file.
rem
rem  It forwards every argument to scripts\start-local.bat, which:
rem    - finds the existing .venv (no activation, no execution policy issues)
rem    - adds the venv's cuDNN 9 directory to PATH for this session
rem      (without that, ctranslate2 silently falls back to CPU after a reboot)
rem    - checks that the port is free, starts uvicorn, opens /docs
rem
rem  Usage:  run.bat                    (or double-click this file)
rem          run.bat -Port 8001
rem          run.bat -Reload -NoBrowser
rem          run.bat -Setup             (only the very first time)
rem
rem  Stop with Ctrl+C.
rem ===========================================================================
call "%~dp0scripts\start-local.bat" %*
exit /b %ERRORLEVEL%
