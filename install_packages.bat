@echo off
echo ======================================================================
echo SRMA-Mamba Package Installation
echo ======================================================================
echo.

echo Installing PyTorch (CPU version)...
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

echo.
echo Installing project requirements...
pip install -r requirements.txt

echo.
echo Installing selective_scan...
cd selective_scan
pip install .
cd ..

echo.
echo Installing Triton (optional)...
pip install triton==2.2.0

echo.
echo ======================================================================
echo Installation complete! Verifying...
echo ======================================================================
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

echo.
echo Now run: python quick_start.py
pause

