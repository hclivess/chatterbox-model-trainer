@echo off
REM Load dataset from speech_splitter_v2.py output

echo ========================================
echo Loading Dataset for Chatterbox Training
echo ========================================
echo.

REM Activate virtual environment
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat first
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

REM Get user input for paths
set /p BASE_DIR="Enter base output directory (e.g., C:/ChatterboxTraining/output): "
set /p SPEAKER_NAME="Enter speaker folder name (e.g., 00 Prolog): "

echo.
echo Loading dataset from: %BASE_DIR%\%SPEAKER_NAME%
echo.

REM Run the dataset loader - THE '--sample-rate 16000' ARGUMENT HAS BEEN REMOVED!
python load_dataset.py ^
  --base-dir "%BASE_DIR%" ^
  --speaker "%SPEAKER_NAME%" ^
  --output "./processed_dataset" ^
  --train-split 0.9

if errorlevel 1 (
    echo.
    echo ERROR: Dataset loading failed!
    echo Check the error message above.\n
    pause
    exit /b 1
)

echo.
echo ========================================
echo Dataset loaded successfully!
echo ========================================
echo.
echo Processed dataset saved to: ./processed_dataset
echo.
echo Next step: Run train.bat to start training
echo.
pause