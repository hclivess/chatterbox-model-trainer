@echo off
REM Chatterbox Training - Quick Setup for Windows
REM This script automates the setup process

echo ========================================
echo Chatterbox Training Setup for Windows
echo RTX 3060 Optimized
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python 3.11 from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo [1/5] Python found
python --version

REM Check if virtual environment exists
if not exist "venv" (
    echo [2/5] Creating virtual environment...
    python -m venv venv
    echo Virtual environment created
) else (
    echo [2/5] Virtual environment already exists
)

REM Activate virtual environment
echo [3/5] Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo [4/5] Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo [5/5] Installing dependencies (this may take a few minutes)...
echo.
echo Installing PyTorch with CUDA 11.8...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo.
echo Installing Chatterbox and training libraries...
pip install chatterbox-tts transformers accelerate

echo.
echo Installing dataset libraries (specific versions for compatibility)...
pip install pyarrow==12.0.1
pip install datasets==2.14.0 pandas

echo.
echo Installing monitoring tools...
pip install tensorboard tqdm

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Make sure your speech_splitter output is in the 'output' folder
echo 2. Run: load_dataset.bat to process your dataset
echo 3. Run: train.bat to start training
echo.
echo Your virtual environment is now ACTIVE.
echo To deactivate it, type: deactivate
echo.
pause
