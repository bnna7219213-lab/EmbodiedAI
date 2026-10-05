@echo off
REM EmbodiedAI one-shot verification (cmd entry) - same logic as scripts\verify.ps1
REM Usage: scripts\verify.bat [--no-pytest]
REM Kept ASCII-only on purpose (GBK bytes in .bat can be misread by cmd).

setlocal
set REPO_ROOT=%~dp0..
pushd "%REPO_ROOT%"

set PYTHONPATH=%REPO_ROOT%\src
set PYTHONIOENCODING=utf-8

echo === EmbodiedAI verify (repo: %REPO_ROOT%) ===
python -m embodied_ai.cli verify %*
set CODE=%ERRORLEVEL%

echo.
if "%CODE%"=="0" (
  echo RESULT=VERIFY_OK
) else (
  echo RESULT=VERIFY_FAIL exit=%CODE%
)
popd
endlocal & exit /b %CODE%
