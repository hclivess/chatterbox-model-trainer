@echo off
REM Chatterbox T3 Fine-tuning - Production Version
REM Supports configurable training parameters and checkpoint resumption

setlocal enabledelayedexpansion

echo.
echo ============================================================
echo    CHATTERBOX T3 FINE-TUNING - PRODUCTION VERSION
echo ============================================================
echo.

REM ========================================
REM Configuration Section
REM ========================================

REM Default training parameters (you can modify these)
set EPOCHS=3
set BATCH_SIZE=1
set GRAD_ACCUM=16
set LEARNING_RATE=5e-5
set SAVE_STEPS=50
set EVAL_STEPS=100
set LOGGING_STEPS=10
set WARMUP_STEPS=100
set RESUME=true

REM Dataset configuration
set DATASET_PATH=processed_dataset
set OUTPUT_DIR=checkpoints

REM Model configuration
set MODEL_NAME_OR_PATH=chatterbox-model
set LOCAL_MODEL_DIR=  REM Set this if using local model files

REM ========================================
REM Pre-flight Checks
REM ========================================

echo [CHECKS] Verifying environment...
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo.
    echo Please run setup.bat first to create the environment.
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment activated

REM Check if CUDA is available
echo [CHECK] Checking CUDA availability...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}'); print(f'Current CUDA device: {torch.cuda.current_device() if torch.cuda.is_available() else \"N/A\"}'); print(f'CUDA device name: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"N/A\"}')" 2>nul
if errorlevel 1 (
    echo [WARNING] Could not check CUDA status. Continuing anyway...
) else (
    echo [OK] CUDA check completed
)
echo.

REM Check if dataset exists
if not exist "%DATASET_PATH%" (
    echo [ERROR] Processed dataset not found!
    echo.
    echo Please run load_dataset.bat first to prepare your dataset.
    echo.
    pause
    exit /b 1
)
echo [OK] Dataset found

REM Check if output directory exists, create if not
if not exist "%OUTPUT_DIR%" (
    mkdir "%OUTPUT_DIR%"
    echo [OK] Created %OUTPUT_DIR% directory
) else (
    echo [OK] %OUTPUT_DIR% directory exists
)

REM Check for existing checkpoints
set CHECKPOINT_EXISTS=false
if exist "%OUTPUT_DIR%\checkpoint-*" (
    set CHECKPOINT_EXISTS=true
    echo [INFO] Found existing checkpoints
)

echo.

REM ========================================
REM Training Configuration Display
REM ========================================

echo ============================================================
echo    TRAINING CONFIGURATION
echo ============================================================
echo.
echo Hardware:
echo   - GPU: RTX 3060 Mobile (6GB VRAM)
echo   - Optimized for memory-constrained training
echo.
echo Training Parameters:
echo   - Epochs: %EPOCHS%
echo   - Batch size: %BATCH_SIZE%
echo   - Gradient accumulation: %GRAD_ACCUM%
echo   - Effective batch size: %BATCH_SIZE% x %GRAD_ACCUM% = !GRAD_ACCUM!
echo   - Learning rate: %LEARNING_RATE%
echo   - Save checkpoint every: %SAVE_STEPS% steps
echo   - Evaluate every: %EVAL_STEPS% steps
echo.
echo Dataset:
echo   - Dataset path: %DATASET_PATH%
echo   - Output directory: %OUTPUT_DIR%
echo.
echo Features:
echo   - FP16: Enabled (performance)
echo   - Gradient checkpointing: Enabled (memory saving)
echo   - Checkpoint resumption: Enabled
echo.

if "%CHECKPOINT_EXISTS%"=="true" (
    echo [INFO] RESUMING FROM PREVIOUS TRAINING
    echo        Your model will continue learning from where it left off
    echo        To start fresh, delete the '%OUTPUT_DIR%' folder
    echo.
)

REM Calculate approximate training time
set /a EFFECTIVE_BATCH=BATCH_SIZE * GRAD_ACCUM
echo Performance Estimates:
echo   - Speed: ~15-20 seconds per training step
echo   - Memory usage: ~4-5GB VRAM
echo.
echo TIP: Monitor GPU in real-time:
echo      Open another terminal and run: nvidia-smi -l 1
echo.
echo ============================================================
echo.

REM ========================================
REM User Confirmation
REM ========================================

echo Ready to start training!
echo.
set /p CONFIRM="Continue with training? (y/n): "
if /i not "!CONFIRM!"=="y" (
    echo.
    echo [CANCELLED] Training cancelled by user.
    echo.
    pause
    exit /b 0
)

REM ========================================
REM Start Training
REM ========================================

echo.
echo ============================================================
echo    STARTING TRAINING
echo ============================================================
echo.
echo Training will begin in 3 seconds...
timeout /t 3 /nobreak >nul
echo.

REM Build the Python command with parameters
set CMD=python finetune_t3.py

REM Add model arguments
if defined LOCAL_MODEL_DIR (
    set CMD=!CMD! --local_model_dir "%LOCAL_MODEL_DIR%"
) else (
    set CMD=!CMD! --model_name_or_path "%MODEL_NAME_OR_PATH%"
)

REM Add data arguments for HuggingFace dataset
set CMD=!CMD! --dataset_name "%DATASET_PATH%" --text_column_name "text" --audio_column_name "audio"

REM Add training arguments - explicitly set GPU usage
set CMD=!CMD! --output_dir "%OUTPUT_DIR%" --num_train_epochs %EPOCHS% --per_device_train_batch_size %BATCH_SIZE% --gradient_accumulation_steps %GRAD_ACCUM% --learning_rate %LEARNING_RATE% --save_steps %SAVE_STEPS% --eval_steps %EVAL_STEPS% --logging_steps %LOGGING_STEPS% --warmup_steps %WARMUP_STEPS% --save_total_limit 2 --save_strategy steps --evaluation_strategy steps --logging_strategy steps --load_best_model_at_end --metric_for_best_model eval_loss --greater_is_better false --fp16 --gradient_checkpointing --dataloader_num_workers 4 --remove_unused_columns false --do_train --do_eval --dataloader_pin_memory true

REM Add resume flag if needed
if "%RESUME%"=="true" (
    if exist "%OUTPUT_DIR%\checkpoint-*" (
        set CMD=!CMD! --resume_from_checkpoint
    )
)

echo [COMMAND] !CMD!
echo.
echo ------------------------------------------------------------
echo.

REM Execute training
%CMD%

REM Capture exit code
set TRAINING_EXIT_CODE=%errorlevel%

REM ========================================
REM Post-Training Summary
REM ========================================

echo.
echo ============================================================
echo    TRAINING SESSION COMPLETED
echo ============================================================
echo.

if %TRAINING_EXIT_CODE% equ 0 (
    echo [SUCCESS] Training completed successfully!
    echo.
    echo Your fine-tuned model is saved in:
    echo   - ./%OUTPUT_DIR%
    echo.
    echo Next steps:
    echo   1. Test your model with generate_tts.bat
    echo   2. Run training again to continue fine-tuning
    echo   3. Share your model or use it in your projects
    echo.
) else (
    echo [WARNING] Training exited with code %TRAINING_EXIT_CODE%
    echo.
    if %TRAINING_EXIT_CODE% equ 1 (
        echo This might be normal if you stopped training with Ctrl+C
        echo Your progress has been saved and you can resume later.
    ) else (
        echo An error may have occurred during training.
        echo Check the error messages above for details.
    )
    echo.
    echo Checkpoints are saved in: ./%OUTPUT_DIR%
    echo.
)

echo ============================================================
echo.

REM ========================================
REM Training Options Menu
REM ========================================

:MENU
echo What would you like to do?
echo.
echo   1. Train again (continue from checkpoint)
echo   2. Train again (start fresh)
echo   3. Test the model
echo   4. Exit
echo.
set /p CHOICE="Enter your choice (1-4): "

if "!CHOICE!"=="1" (
    echo.
    echo Restarting training with checkpoint resumption...
    echo.
    timeout /t 2 /nobreak >nul
    cls
    goto :START_TRAINING
)

if "!CHOICE!"=="2" (
    echo.
    echo [WARNING] This will start training from scratch!
    set /p CONFIRM_FRESH="Are you sure? Previous training will be overwritten (y/n): "
    if /i "!CONFIRM_FRESH!"=="y" (
        echo.
        echo Deleting checkpoints...
        rd /s /q "%OUTPUT_DIR%" 2>nul
        mkdir "%OUTPUT_DIR%"
        echo.
        echo Starting fresh training...
        timeout /t 2 /nobreak >nul
        cls
        goto :START_TRAINING
    ) else (
        echo.
        echo Cancelled.
        goto :MENU
    )
)

if "!CHOICE!"=="3" (
    echo.
    echo Opening TTS generation script...
    if exist "generate_tts.bat" (
        call generate_tts.bat
    ) else (
        echo [ERROR] generate_tts.bat not found
        echo Please create this script to test your model
    )
    echo.
    goto :MENU
)

if "!CHOICE!"=="4" (
    goto :END
)

echo Invalid choice. Please try again.
echo.
goto :MENU

:START_TRAINING
REM Jump back to training section
goto :eof

:END
echo.
echo Goodbye!
echo.
pause
exit /b 0
