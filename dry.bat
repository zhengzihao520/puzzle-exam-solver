@echo off
cd /d "%~dp0"
echo.
echo   READ-ONLY: it looks at the screen and prints the plan. No mouse action.
echo   Start the game yourself; this waits for the deal, then prints what it would do.
echo.
D:\python\python.exe stateless.py --dry --wait 8 --wait-deal 600
echo.
echo   exit code: %ERRORLEVEL%
pause
