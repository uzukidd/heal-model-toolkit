@echo off
setlocal
pushd "%~dp0.." || exit /b 1
if defined HEAL_WSL_DISTRO (
    wsl.exe --distribution "%HEAL_WSL_DISTRO%" --exec bash scripts/bevfusion_board_demo.sh %*
) else (
    wsl.exe --exec bash scripts/bevfusion_board_demo.sh %*
)
set "RESULT=%ERRORLEVEL%"
popd
exit /b %RESULT%
