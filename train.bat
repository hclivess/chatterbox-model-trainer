@echo off
REM Start Chatterbox training

echo ========================================
echo Chatterbox Training - RTX 3060 Optimized
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

REM Check if dataset exists
if not exist "processed_dataset" (
    echo ERROR: Processed dataset not found!
    echo Please run load_dataset.bat first
    pause
    exit /b 1
)

echo Training Configuration:
echo - Batch size: 1 (small for 6GB VRAM)
echo - Gradient accumulation: 16 (your friend's advice!)
echo - Effective batch size: 16
echo - Epochs: 3
echo - FP16: Enabled
echo - Gradient checkpointing: Enabled
echo.
echo WARNING: Training will be SLOW on RTX 3060 mobile!
echo Expected: 10-20 seconds per step
echo.
echo TIP: Open another PowerShell and run "nvidia-smi -l 1"
echo      to monitor GPU memory usage in real-time
echo.

set /p CONFIRM="Start training? (y/n): "
if /i not "%CONFIRM%"=="y" (
    echo Training cancelled.
    pause
    exit /b 0
)

echo.
echo Starting training...
echo.

REM Start training with correct arguments
python train_chatterbox.py --dataset "./processed_dataset" --output "./checkpoints"

echo.
echo ========================================
echo Training session ended
echo ========================================
echo.
pause
