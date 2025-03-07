@echo off
REM process_file.bat - Windows batch script for processing files with Smart Organizer

REM Check if Python is installed and in the PATH
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python is not installed or not in your PATH.
    echo Please install Python and make sure it's in your PATH.
    pause
    exit /b 1
)

REM Get the file path from drag-and-drop or command line argument
set "file_path=%~1"

REM If no argument is provided, prompt for a file path
if "%file_path%"=="" (
    set /p file_path="Enter the file path to process: "
)

REM Check if the file exists
if not exist "%file_path%" (
    echo Error: File not found.
    pause
    exit /b 1
)

REM Execute the smart_organizer.py script with the file path
python "%~dp0smart_organizer.py" file "%file_path%"

REM Pause to see results
pause
