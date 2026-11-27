@echo off
setlocal
pushd "%~dp0.." || exit /b 1
if defined HEAL_WSL_DISTRO (
    wsl.exe --distribution "%HEAL_WSL_DISTRO%" --exec bash scripts/compile_bevfusion_s600.sh %*
) else (
    wsl.exe --exec bash scripts/compile_bevfusion_s600.sh %*
)
set "RESULT=%ERRORLEVEL%"
popd
exit /b %RESULT%
