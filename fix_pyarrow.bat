@echo off
REM Quick fix for PyArrow compatibility issue
REM Run this if you get: AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'

echo ========================================
echo Quick Fix: PyArrow Compatibility Issue
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

echo Fixing PyArrow version conflict...
echo.

REM Uninstall incompatible versions
echo [1/3] Uninstalling current pyarrow...
pip uninstall pyarrow -y

echo.
echo [2/3] Installing compatible pyarrow version (12.0.1)...
pip install pyarrow==12.0.1

echo.
echo [3/3] Reinstalling datasets library...
pip install datasets==2.14.0 --force-reinstall --no-deps
pip install pandas tqdm pyarrow==12.0.1

echo.
echo ========================================
echo Fix Complete!
echo ========================================
echo.
echo The PyArrow compatibility issue should now be resolved.
echo You can now run load_dataset.bat
echo.
pause
