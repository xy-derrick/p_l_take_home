@echo off
setlocal
python "%~dp0build_ava_manifest.py" %*
set EXITCODE=%ERRORLEVEL%
endlocal & exit /b %EXITCODE%
