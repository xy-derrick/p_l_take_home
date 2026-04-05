@echo off
setlocal
python "%~dp0generate_corruption_example.py" %*
set EXITCODE=%ERRORLEVEL%
endlocal & exit /b %EXITCODE%
