@echo off
setlocal
python "%~dp0download_ava_subset.py" %*
set EXITCODE=%ERRORLEVEL%
endlocal & exit /b %EXITCODE%
