@echo off
cd /d "%~dp0"
echo.
echo   ==================================================================
echo   1. Put the exam page in front, maximised.
echo   2. Start the game yourself: choose the part, 25 cells (5x5), start.
echo      This program does NOT click the setup screen. It only waits.
echo   3. It takes over as soon as the deal appears, and stops at 25/25.
echo   ==================================================================
echo.
D:\python\python.exe stateless.py --go --wait 8 --wait-deal 900
echo.
echo   exit code: %ERRORLEVEL%   (0 = solved, 2 = wrong foreground, 3 = no 5x5 deal seen)
pause
