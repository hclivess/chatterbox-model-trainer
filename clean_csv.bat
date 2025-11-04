@echo off
REM Clean up metadata.csv after deleting audio files

echo ========================================
echo Metadata.csv Cleanup Tool
echo ========================================
echo.
echo This tool removes entries from metadata.csv for
echo audio files that you've deleted.
echo.
echo It will:
echo  1. Check which audio files are missing
echo  2. Create a backup of your metadata.csv
echo  3. Remove entries for missing files
echo.

REM Activate virtual environment
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat first
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

REM Get user input
set /p BASE_DIR="Enter base output directory (e.g., C:/ChatterboxTraining/output): "
set /p SPEAKER_NAME="Enter speaker folder name (e.g., 00 Prolog): "

echo.
echo Checking: %BASE_DIR%\%SPEAKER_NAME%
echo.

REM Run cleanup
python clean_csv.py ^
  --base-dir "%BASE_DIR%" ^
  --speaker "%SPEAKER_NAME%"

if errorlevel 1 (
    echo.
    echo ERROR: Cleanup failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Cleanup Complete!
echo ========================================
echo.
echo Your metadata.csv is now in sync with your audio files.
echo You can now run load_dataset.bat
echo.
pause
