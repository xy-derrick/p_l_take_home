@echo off
setlocal
python "%~dp0build_greatest_hits_manifest.py" %*
set EXITCODE=%ERRORLEVEL%
endlocal & exit /b %EXITCODE%
