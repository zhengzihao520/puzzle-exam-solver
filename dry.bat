@echo off
cd /d "%~dp0"
echo.
echo   READ-ONLY: it looks at the screen and prints the plan. No mouse action.
echo   Click the exam window now -- it must be maximised and in front.
echo.
D:\python\python.exe stateless.py --dry --wait 10
echo.
echo   exit code: %ERRORLEVEL%
pause
