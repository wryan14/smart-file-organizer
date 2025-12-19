# Install-SmartOrganizer.ps1 - PowerShell script to set up Smart Organizer

# Define the installation directory
$installDir = "$env:USERPROFILE\SmartOrganizer"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Create installation directory if it doesn't exist
if (-not (Test-Path $installDir)) {
    Write-Host "Creating installation directory..." -ForegroundColor Cyan
    New-Item -ItemType Directory -Path $installDir | Out-Null
}

# Copy the Python script and requirements
Write-Host "Installing Smart Organizer..." -ForegroundColor Cyan
Copy-Item "$scriptDir\smart_organizer.py" -Destination "$installDir\smart_organizer.py" -Force
Copy-Item "$scriptDir\requirements.txt" -Destination "$installDir\requirements.txt" -Force

# Copy the batch files
Copy-Item "$scriptDir\process_file.bat" -Destination "$installDir\process_file.bat" -Force
Copy-Item "$scriptDir\process_directory.bat" -Destination "$installDir\process_directory.bat" -Force
Copy-Item "$scriptDir\batch_process.bat" -Destination "$installDir\batch_process.bat" -Force

# Create desktop shortcuts
$WshShell = New-Object -ComObject WScript.Shell

# File processing shortcut
$fileShortcut = $WshShell.CreateShortcut("$env:USERPROFILE\Desktop\Process File.lnk")
$fileShortcut.TargetPath = "$installDir\process_file.bat"
$fileShortcut.WorkingDirectory = $installDir
$fileShortcut.IconLocation = "shell32.dll,0"
$fileShortcut.Description = "Process a file with Smart Organizer"
$fileShortcut.Save()

# Directory processing shortcut
$dirShortcut = $WshShell.CreateShortcut("$env:USERPROFILE\Desktop\Process Directory.lnk")
$dirShortcut.TargetPath = "$installDir\process_directory.bat"
$dirShortcut.WorkingDirectory = $installDir
$dirShortcut.IconLocation = "shell32.dll,3"
$dirShortcut.Description = "Process a directory with Smart Organizer"
$dirShortcut.Save()

# Batch processing shortcut
$batchShortcut = $WshShell.CreateShortcut("$env:USERPROFILE\Desktop\Batch Process.lnk")
$batchShortcut.TargetPath = "$installDir\batch_process.bat"
$batchShortcut.WorkingDirectory = $installDir
$batchShortcut.IconLocation = "shell32.dll,46"
$batchShortcut.Description = "Batch process multiple files with Smart Organizer"
$batchShortcut.Save()

# Check if Python and required packages are installed
try {
    Write-Host "Checking Python installation..." -ForegroundColor Cyan
    $pythonVersion = python --version 2>&1
    
    if ($pythonVersion -match "Python (3\.\d+)") {
        Write-Host "Python $($matches[1]) is installed." -ForegroundColor Green
        
        # Install required packages from requirements.txt
        Write-Host "Installing required packages..." -ForegroundColor Cyan
        python -m pip install -q -r "$installDir\requirements.txt"
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Required packages installed successfully." -ForegroundColor Green
        } else {
            Write-Host "Warning: Some packages may not have installed correctly." -ForegroundColor Yellow
        }
    } else {
        Write-Host "Python 3.x is not installed or not in your PATH." -ForegroundColor Red
        Write-Host "Please install Python 3.6 or higher from https://www.python.org/" -ForegroundColor Yellow
    }
} catch {
    Write-Host "Error checking Python installation: $_" -ForegroundColor Red
    Write-Host "Please install Python 3.6 or higher from https://www.python.org/" -ForegroundColor Yellow
}

# Create environment file for API key
$envPath = "$installDir\.env"
if (-not (Test-Path $envPath)) {
    Write-Host "`nSetting up API key..." -ForegroundColor Cyan
    Write-Host "Get an API key at https://openrouter.ai/" -ForegroundColor Cyan
    $apiKey = Read-Host "Enter your OpenRouter API key (or press Enter to skip)"

    if ($apiKey) {
        "OPENROUTER_API_KEY=$apiKey" | Out-File -FilePath $envPath
        Write-Host "API key saved to $envPath" -ForegroundColor Green
    } else {
        Write-Host "Skipped API key setup. You'll need to set OPENROUTER_API_KEY later." -ForegroundColor Yellow
    }
}

Write-Host "`nSmartOrganizer installed successfully!" -ForegroundColor Green
Write-Host "You can find shortcuts on your desktop:" -ForegroundColor Cyan
Write-Host "  - Process File - For processing individual files" -ForegroundColor Cyan
Write-Host "  - Process Directory - For processing entire directories" -ForegroundColor Cyan
Write-Host "  - Batch Process - For efficient processing of multiple files" -ForegroundColor Cyan
Write-Host "`nYou can also run the tool from command line:" -ForegroundColor Cyan
Write-Host "  python $installDir\smart_organizer.py [command] [arguments]" -ForegroundColor Cyan

# Pause to see results
Write-Host "`nPress any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
