@echo off
cd /d C:\Users\siddh\Desktop\PBL
set TRIES=0
:retry
echo [%date% %time%] collect attempt >> data\interim\collect_run.log
.venv\Scripts\python.exe src\live_pipeline.py collect >> data\interim\collect_run.log 2>&1
if %ERRORLEVEL% NEQ 2 goto done
set /a TRIES+=1
if %TRIES% GEQ 6 goto done
ping -n 601 127.0.0.1 >nul
goto retry
:done