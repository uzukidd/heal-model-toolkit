@echo off
setlocal
pushd "%~dp0.." || exit /b 1
if defined HEAL_WSL_DISTRO (
    wsl.exe --distribution "%HEAL_WSL_DISTRO%" --exec bash scripts/run_bevfusion_float.sh %*
) else (
    wsl.exe --exec bash scripts/run_bevfusion_float.sh %*
)
set "RESULT=%ERRORLEVEL%"
popd
exit /b %RESULT%
