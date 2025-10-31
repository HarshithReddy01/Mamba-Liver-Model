# Environment Setup Guide for SRMA-Mamba

## 🚀 Quick Setup (Automated)

**Option 1: Use the setup script (Recommended)**
```bash
python setup_environment.py
```
This interactive script will guide you through the setup process.

**Option 2: Manual setup (see below)**

---

## 📋 Prerequisites

- Windows 10/11 (or Linux/Mac)
- Python 3.9
- CUDA-capable GPU (optional but recommended)
- 10GB+ free disk space

---

## 🔧 Step-by-Step Manual Setup

### Step 1: Install Conda (if not installed)

**Windows:**
1. Download Miniconda: https://docs.conda.io/en/latest/miniconda.html
2. Run installer and follow prompts
3. Restart terminal/PowerShell

**Verify:**
```bash
conda --version
```

### Step 2: Create Conda Environment

```bash
# Create environment with Python 3.9
conda create -n SRMA-Mamba python=3.9.0 -y

# Activate environment
conda activate SRMA-Mamba
```

**Verify:**
```bash
python --version  # Should show Python 3.9.x
```

### Step 3: Install PyTorch with CUDA

**For GPU (CUDA 11.8):**
```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

**For CPU only:**
```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
```

**Verify CUDA:**
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
```

### Step 4: Install Project Dependencies

```bash
# Make sure you're in the project directory
cd "D:\Fall 2025\RA work\Liver Cirr Model"

# Install requirements
pip install -r requirements.txt
```

### Step 5: Install Selective Scan (CUDA kernels)

```bash
cd selective_scan
pip install .
cd ..
```

### Step 6: Install Triton (Optional but Recommended)

```bash
pip install triton==2.2.0
```

**Note:** Triton may require specific CUDA versions. If installation fails, you can skip this step - the model will still work.

### Step 7: Verify Installation

```bash
python quick_start.py
```

This will check all packages and verify everything is working.

---

## ✅ Verification Checklist

After setup, verify:

- [ ] Python 3.9 installed
- [ ] Conda environment activated
- [ ] PyTorch installed and CUDA working (if GPU available)
- [ ] All packages from requirements.txt installed
- [ ] selective_scan installed
- [ ] `quick_start.py` runs without errors
- [ ] Model can be initialized: `python configs/model_configs.py`

---

## 🐛 Common Issues & Solutions

### Issue 1: "conda: command not found"

**Solution:**
- Install Miniconda/Anaconda
- Restart terminal
- Add conda to PATH if needed

### Issue 2: CUDA not available after installing PyTorch

**Solution:**
1. Check NVIDIA driver: `nvidia-smi`
2. Verify CUDA toolkit is installed
3. Reinstall PyTorch with correct CUDA version:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
   ```

### Issue 3: selective_scan installation fails

**Solution:**
- Ensure you're in the `selective_scan` directory
- Check CUDA toolkit is installed
- Try: `pip install . --no-build-isolation`

### Issue 4: "ModuleNotFoundError" for mamba_ssm

**Solution:**
```bash
pip install mamba-ssm==2.2.2
```

### Issue 5: Triton installation fails

**Solution:**
- This is optional - model will work without it
- Skip this step if you get errors
- Or try: `pip install triton --upgrade`

---

## 📝 Environment Variables (Optional)

You may want to set these for convenience:

**Windows PowerShell:**
```powershell
$env:CUDA_VISIBLE_DEVICES = "0"
```

**Windows CMD:**
```cmd
set CUDA_VISIBLE_DEVICES=0
```

**Linux/Mac:**
```bash
export CUDA_VISIBLE_DEVICES=0
```

---

## 🧪 Quick Test

After setup, test everything:

```bash
# 1. Activate environment
conda activate SRMA-Mamba

# 2. Verify packages
python quick_start.py

# 3. Test model initialization
python configs/model_configs.py
```

Expected output from `configs/model_configs.py`:
```
tensor([[[[[...]]]]])  # Tensor shapes printed
```

---

## 📦 Package Versions

**Critical packages:**
- Python: 3.9.0
- PyTorch: 2.0.1
- MONAI: 1.4.0
- mamba-ssm: 2.2.2

See `requirements.txt` for full list.

---

## 🔄 Updating Environment

If you need to update packages:

```bash
# Activate environment
conda activate SRMA-Mamba

# Update requirements
pip install -r requirements.txt --upgrade

# Reinstall selective_scan if updated
cd selective_scan
pip install . --force-reinstall
cd ..
```

---

## 💾 Environment Management

**List environments:**
```bash
conda env list
```

**Deactivate environment:**
```bash
conda deactivate
```

**Remove environment (if needed):**
```bash
conda env remove -n SRMA-Mamba
```

---

## 🎯 Next Steps After Setup

1. ✅ Verify setup: `python quick_start.py`
2. 📥 Download dataset (if not already done)
3. 🚂 Train model: `python train.py`
4. 🧪 Test model: `python test_model.py`

---

## 📞 Still Having Issues?

1. **Check logs:** Look at error messages carefully
2. **Verify versions:** Ensure package versions match requirements.txt
3. **Clean install:** Remove environment and start fresh:
   ```bash
   conda deactivate
   conda env remove -n SRMA-Mamba
   # Then start from Step 2 again
   ```

---

**Ready? Run:** `python setup_environment.py` to get started! 🚀

