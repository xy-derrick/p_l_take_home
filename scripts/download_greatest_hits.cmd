@echo off
setlocal
python "%~dp0download_greatest_hits.py" %*
set EXITCODE=%ERRORLEVEL%
endlocal & exit /b %EXITCODE%
