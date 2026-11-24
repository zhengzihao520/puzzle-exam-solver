@echo off
cd /d "%~dp0"
echo.
echo   ================================================================
echo    1. The exam page must be open, MAXIMISED and in front.
echo    2. You have 10 seconds to click on it after this line.
echo    3. It stops by itself when the board reaches 25/25.
echo   ================================================================
echo.
D:\python\python.exe stateless.py --go --wait 10
echo.
echo   exit code: %ERRORLEVEL%   (0 = done, 2 = bad foreground, other = see log above)
pause
