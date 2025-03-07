@echo off
REM process_directory.bat - Windows batch script for processing directories with Smart Organizer

REM Check if Python is installed and in the PATH
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python is not installed or not in your PATH.
    echo Please install Python and make sure it's in your PATH.
    pause
    exit /b 1
)

REM Get the directory path from drag-and-drop or command line argument
set "dir_path=%~1"

REM If no argument is provided, prompt for a directory path
if "%dir_path%"=="" (
    set /p dir_path="Enter the directory path to process: "
)

REM Check if the directory exists
if not exist "%dir_path%\" (
    echo Error: Directory not found.
    pause
    exit /b 1
)

REM Execute the smart_organizer.py script with the directory path
python "%~dp0smart_organizer.py" dir "%dir_path%"

REM Pause to see results
pause
