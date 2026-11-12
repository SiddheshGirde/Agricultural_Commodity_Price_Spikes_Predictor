@echo off
rem Daily collector. Works from any location: it moves to the project root (the folder above scripts\).
cd /d "%~dp0.."
if not exist data\interim mkdir data\interim
set TRIES=0
:retry
echo [%date% %time%] collect attempt >> data\interim\collect_run.log
".venv\Scripts\python.exe" src\live_pipeline.py collect >> data\interim\collect_run.log 2>&1
rem exit code 2 = the data source was unreachable and today is still open: wait 10 minutes, up to 6 tries
if %ERRORLEVEL% NEQ 2 goto done
set /a TRIES+=1
if %TRIES% GEQ 6 goto done
ping -n 601 127.0.0.1 >nul
goto retry
:done