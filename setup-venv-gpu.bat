@echo off

setlocal

title OSEDiff - crear .venv con GPU

cd /d "%~dp0"



echo.

echo  Crea .venv e instala dependencias + PyTorch CUDA 11.8 (torch 2.0.1)

echo  Tras esto usa run_api.bat o PM2-start.bat

echo.



where python >nul 2>&1

if errorlevel 1 (

    echo  Python no esta en el PATH.

    pause

    exit /b 1

)



if not exist ".venv\Scripts\python.exe" (

    echo  Creando .venv...

    python -m venv .venv

)



set "PIP=.venv\Scripts\pip.exe"

set "PYTORCH_INDEX=https://download.pytorch.org/whl/cu118"



echo  requirements.txt...

call "%PIP%" install -r requirements.txt



echo  PyTorch CUDA (al final, para no quedar con build CPU)...

call "%PIP%" uninstall -y torch torchvision

call "%PIP%" install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url %PYTORCH_INDEX%

call "%PIP%" install "numpy==1.26.4" "Pillow==9.5.0"



echo.

".venv\Scripts\python.exe" -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"

echo.

pause

exit /b 0


