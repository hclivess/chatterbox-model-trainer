@echo off
REM Setup full Chatterbox fine-tuning

echo ========================================
echo Chatterbox FULL Fine-tuning Setup
echo ========================================
echo.
echo This will set you up for the "full experience"
echo Not LoRA - full model fine-tuning!
echo.
echo WARNING: Requires 6GB VRAM minimum
echo          Your RTX 3060 mobile is at the limit
echo          Training will be VERY slow but possible
echo.

REM Activate venv
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Run setup.bat first
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

REM Check if chatterbox-finetuning exists
if exist "chatterbox-finetuning" (
    echo Found existing chatterbox-finetuning directory
    set /p RECLONE="Re-clone from GitHub? (y/n): "
    if /i "%RECLONE%"=="y" (
        echo Removing old directory...
        rmdir /s /q chatterbox-finetuning
        goto :clone
    )
    goto :install_deps
)

:clone
echo.
echo [1/3] Cloning chatterbox-finetuning repository...
git clone https://github.com/stlohrey/chatterbox-finetuning.git

if errorlevel 1 (
    echo.
    echo ERROR: Failed to clone repository
    echo Make sure Git is installed
    pause
    exit /b 1
)

:install_deps
echo.
echo [2/3] Installing dependencies...
cd chatterbox-finetuning

if exist "requirements.txt" (
    pip install -r requirements.txt
) else (
    echo No requirements.txt found, installing base requirements...
    pip install chatterbox-tts transformers accelerate datasets
)

cd ..

echo.
echo [3/3] Creating training script...
REM train_full.py is already created

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Repository cloned to: .\chatterbox-finetuning\
echo.
echo NEXT STEPS:
echo 1. Review their training examples in chatterbox-finetuning\
echo 2. Run: python train_full.py
echo 3. Follow the integration instructions shown
echo.
echo Your dataset (42 train, 5 val) is ready to use!
echo.
pause
