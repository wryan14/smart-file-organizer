@echo off
REM batch_process.bat - Windows batch script for batch processing with Smart Organizer

REM Check if Python is installed and in the PATH
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python is not installed or not in your PATH.
    echo Please install Python and make sure it's in your PATH.
    pause
    exit /b 1
)

REM Execute the smart_organizer.py script in batch mode
python "%~dp0smart_organizer.py" batch

REM Pause to see results
pause
