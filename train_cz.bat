@echo off
setlocal enabledelayedexpansion

REM ========================================
REM Configuration
REM ========================================
set EPOCHS=3
set BATCH_SIZE=1
set GRAD_ACCUM=16
set LEARNING_RATE=5e-5
set SAVE_STEPS=50
set EVAL_STEPS=100
set LOGGING_STEPS=10
set WARMUP_STEPS=100
set DATASET_PATH=processed_dataset
set OUTPUT_DIR=chatterbox-czech-finetuned

REM Czech model is a T3-only checkpoint on top of base Chatterbox
REM We need to download base model + Czech T3 weights
set BASE_MODEL=ResembleAI/chatterbox
set CZECH_T3_REPO=Thomcles/chatterbox-Czech
set CZECH_T3_FILE=t3_cs.safetensors

REM Dataset column names
set TEXT_COLUMN=text
set AUDIO_COLUMN=audio
set TRAIN_SPLIT=train

REM ========================================
REM Checks
REM ========================================
echo Checking environment...

if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found! Run setup.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

if not exist "%DATASET_PATH%" (
    echo ERROR: Dataset not found at %DATASET_PATH%
    echo Run load_dataset.bat first.
    pause
    exit /b 1
)

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo.
echo ============================================================
echo   CHATTERBOX CZECH T3 FINE-TUNING SETUP
echo ============================================================
echo.
echo This script will:
echo   1. Download base Chatterbox model (~2GB)
echo   2. Download Czech T3 weights (~2GB)  
echo   3. Merge them to create Czech base model
echo   4. Fine-tune on your dataset
echo.
echo NOTE: You need HuggingFace authentication for gated repos
echo       Run: huggingface-cli login
echo.

REM Check if already logged in
python -c "from huggingface_hub import HfFolder; token = HfFolder.get_token(); exit(0 if token else 1)" 2>nul
if errorlevel 1 (
    echo WARNING: Not logged in to HuggingFace
    echo.
    echo Please run: huggingface-cli login
    echo Then run this script again.
    echo.
    pause
    exit /b 1
)

echo ✓ HuggingFace authentication detected
echo.
echo Config:
echo   Epochs: %EPOCHS%
echo   Batch: %BATCH_SIZE% x %GRAD_ACCUM% = %GRAD_ACCUM% effective
echo   Learning rate: %LEARNING_RATE%
echo   Base model: %BASE_MODEL%
echo   Czech T3: %CZECH_T3_REPO%/%CZECH_T3_FILE%
echo   Dataset: %DATASET_PATH%
echo   Output: %OUTPUT_DIR%
echo.

set /p CONFIRM="Continue with training? (y/n): "
if /i not "!CONFIRM!"=="y" (
    echo Cancelled.
    pause
    exit /b 0
)

REM ========================================
REM Download and Setup Czech Model
REM ========================================
echo.
echo ============================================================
echo STEP 1: Preparing Czech base model...
echo ============================================================

set CZECH_BASE_DIR=czech_base_model
if not exist "%CZECH_BASE_DIR%" mkdir "%CZECH_BASE_DIR%"

echo Downloading base Chatterbox components...
python -c "from huggingface_hub import hf_hub_download; import sys; files = ['ve.safetensors', 't3_cfg.safetensors', 's3gen.safetensors', 'tokenizer.json', 'conds.pt']; [hf_hub_download('%BASE_MODEL%', f, local_dir='%CZECH_BASE_DIR%', local_dir_use_symlinks=False) for f in files]; print('✓ Base model downloaded')"

if errorlevel 1 (
    echo ERROR: Failed to download base model
    pause
    exit /b 1
)

echo.
echo Downloading Czech T3 weights...
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('%CZECH_T3_REPO%', '%CZECH_T3_FILE%', local_dir='%CZECH_BASE_DIR%', local_dir_use_symlinks=False); print('✓ Czech T3 downloaded')"

if errorlevel 1 (
    echo ERROR: Failed to download Czech T3 weights
    pause
    exit /b 1
)

echo.
echo Replacing base T3 with Czech T3...
copy /Y "%CZECH_BASE_DIR%\%CZECH_T3_FILE%" "%CZECH_BASE_DIR%\t3_cfg.safetensors" >nul

if errorlevel 1 (
    echo ERROR: Failed to copy Czech T3 weights
    pause
    exit /b 1
)

echo ✓ Czech base model ready at: %CZECH_BASE_DIR%

REM ========================================
REM Training
REM ========================================
echo.
echo ============================================================
echo STEP 2: Fine-tuning Czech model on your data...
echo ============================================================
echo.
echo Starting training in 3 seconds...
timeout /t 3 /nobreak >nul

echo.
echo ============================================================
echo Python output:
echo ============================================================

python -u finetune_t3.py ^
  --local_model_dir "%CZECH_BASE_DIR%" ^
  --dataset_name "%DATASET_PATH%" ^
  --train_split_name %TRAIN_SPLIT% ^
  --text_column_name %TEXT_COLUMN% ^
  --audio_column_name %AUDIO_COLUMN% ^
  --output_dir "%OUTPUT_DIR%" ^
  --num_train_epochs %EPOCHS% ^
  --per_device_train_batch_size %BATCH_SIZE% ^
  --gradient_accumulation_steps %GRAD_ACCUM% ^
  --learning_rate %LEARNING_RATE% ^
  --save_steps %SAVE_STEPS% ^
  --eval_strategy steps ^
  --eval_steps %EVAL_STEPS% ^
  --logging_steps %LOGGING_STEPS% ^
  --warmup_steps %WARMUP_STEPS% ^
  --save_total_limit 2 ^
  --save_strategy steps ^
  --load_best_model_at_end ^
  --metric_for_best_model eval_loss ^
  --fp16 ^
  --gradient_checkpointing ^
  --dataloader_num_workers 4 ^
  --do_train ^
  --do_eval ^
  --ignore_verifications ^
  --resume_from_checkpoint

set EXIT_CODE=%errorlevel%

echo.
echo ============================================================
if %EXIT_CODE% equ 0 (
    echo SUCCESS! Model saved to %OUTPUT_DIR%
    echo.
    echo The following files have been created:
    echo   - t3_cfg.safetensors (your finetuned Czech T3 model)
    echo   - ve.safetensors (voice encoder)
    echo   - s3gen.safetensors (speech generator)
    echo   - tokenizer.json (tokenizer)
    echo.
    echo Usage:
    echo   from chatterbox.tts import ChatterboxTTS
    echo   model = ChatterboxTTS.from_local("%OUTPUT_DIR%", device="cuda")
    echo   wav = model.generate("Váš český text zde")
) else (
    echo Training failed with error code %EXIT_CODE%
    echo Check the output above for error messages
)
echo ============================================================
echo.
pause