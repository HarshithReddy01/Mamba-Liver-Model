# Environment Setup Status

## ✅ Successfully Installed

- ✅ PyTorch 2.9.0
- ✅ MONAI 1.5.1
- ✅ NumPy, Pandas, OpenCV
- ✅ Transformers, Timm, Fvcore
- ✅ Nibabel, TQDM, Scikit-learn
- ✅ All other core dependencies

## ❌ Still Need

1. **mamba-ssm** - Installation failing (Python 3.12 compatibility issue)
2. **Dataset** - Not downloaded yet
3. **selective_scan** - Needs to be built

---

## 🎯 Solution: Use Conda Environment (Recommended)

The conda environment `SRMA-Mamba` was created with Python 3.9 (which is compatible). 

### Steps:

1. **Open a NEW terminal/PowerShell**

2. **Activate conda environment:**
   ```bash
   conda activate SRMA-Mamba
   ```

3. **Verify Python version:**
   ```bash
   python --version  # Should show 3.9.x
   ```

4. **Install packages in conda environment:**
   ```bash
   cd "D:\Fall 2025\RA work\Liver Cirr Model"
   
   # Install PyTorch
   pip install torch torchvision torchaudio
   
   # Install requirements
   pip install monai pyyaml yacs nibabel opencv-python pandas tqdm scikit-learn
   pip install huggingface-hub tokenizers transformers timm fvcore
   
   # Install mamba-ssm (should work with Python 3.9)
   pip install mamba-ssm
   
   # Build selective_scan
   cd selective_scan
   pip install .
   cd ..
   ```

5. **Verify:**
   ```bash
   python quick_start.py
   ```

---

## 🚀 Alternative: Continue with Current Setup

If you want to proceed with Python 3.12:

### Option 1: Skip mamba-ssm (if model has fallback)
Check if model can work without it by testing:
```bash
python -c "from configs.model_configs import build_SRMAMamba; print('Works!')"
```

### Option 2: Download Dataset First
While mamba-ssm is being sorted, you can:
1. Download dataset from: https://osf.io/cuk24/files/osfstorage
2. Organize in `data/Cirrhosis_T2_3D/` structure

### Option 3: Wait for mamba-ssm Python 3.12 Support
Check if there's a newer version or workaround

---

## 📋 Current Priority Checklist

- [ ] **Activate conda environment** (`conda activate SRMA-Mamba`)
- [ ] **Install packages in conda env** (see commands above)
- [ ] **Download dataset** (https://osf.io/cuk24/files/osfstorage)
- [ ] **Verify setup** (`python quick_start.py`)
- [ ] **Test model** (`python test_model.py` - will need checkpoint)

---

## 💡 Recommendation

**Use the conda environment with Python 3.9** - it's already created and will avoid compatibility issues with mamba-ssm and other packages that expect Python 3.9.

---

**Next Command to Run:**
```bash
conda activate SRMA-Mamba
```

Then continue with package installation in that environment.

