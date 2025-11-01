# Running SRMA-Mamba in Google Colab

## ✅ Why Colab is Perfect for This Project

- ✅ Free GPU access (T4 GPU)
- ✅ Python 3.9+ pre-installed
- ✅ CUDA already configured
- ✅ Most packages available
- ✅ Easy file upload/download
- ✅ No local setup headaches!

---

## 🚀 Quick Start in Colab

### Step 1: Create New Colab Notebook

1. Go to: https://colab.research.google.com/
2. Click **"New Notebook"**
3. Enable GPU: **Runtime → Change runtime type → GPU → Save**

### Step 2: Clone/Upload Your Project

**Option A: Clone from GitHub (if you have it there)**
```python
!git clone https://your-repo-url.git
%cd your-repo-name
```

**Option B: Upload Project Files**
1. Upload the project folder to Google Drive
2. Mount Drive in Colab:
```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/path/to/Liver\ Cirr\ Model
```

**Option C: Upload Directly to Colab**
Use the file browser to upload project files

### Step 3: Install Dependencies

Copy and run this in a Colab cell:

```python
# Install PyTorch with CUDA
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
!pip install monai pyyaml yacs nibabel opencv-python pandas tqdm scikit-learn
!pip install huggingface-hub tokenizers transformers timm fvcore
!pip install mamba-ssm

# Install selective_scan (if available)
%cd selective_scan
!pip install .
%cd ..

print("✅ All packages installed!")
```

### Step 4: Upload Dataset

**Option A: Upload to Google Drive**
1. Upload dataset to Drive
2. Create symlink or copy:
```python
# If dataset is in Drive
!cp -r /content/drive/MyDrive/path/to/Cirrhosis_T2_3D /content/drive/MyDrive/path/to/Liver\ Cirr\ Model/data/
```

**Option B: Direct Upload**
```python
# Use Colab file browser or:
from google.colab import files
# Then upload dataset files
```

**Option C: Download from OSF**
```python
# Download dataset directly
!wget -O dataset.zip "https://osf.io/download/your-dataset-link"
!unzip dataset.zip
!mkdir -p data
!mv Cirrhosis_T2_3D data/
```

### Step 5: Verify Setup

```python
# Run verification
exec(open('quick_start.py').read())
```

Or run individual checks:
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Test model
from configs.model_configs import build_SRMAMamba
model = build_SRMAMamba().cuda()
print("✅ Model loaded successfully!")
```

### Step 6: Train the Model

```python
# Train on Colab GPU
exec(open('train.py').read())
```

Or run directly:
```python
!python train.py
```

---

## 📝 Complete Colab Notebook Template

Here's a complete notebook you can use:

```python
# ============================================
# SRMA-Mamba Setup and Training in Colab
# ============================================

# Step 1: Enable GPU
# Runtime → Change runtime type → GPU → Save

# Step 2: Install dependencies
print("Installing packages...")
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install monai pyyaml yacs nibabel opencv-python pandas tqdm scikit-learn
!pip install huggingface-hub tokenizers transformers timm fvcore
!pip install mamba-ssm

# Step 3: Upload your project files
# Use file browser or:
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/path/to/Liver\ Cirr\ Model

# Step 4: Build selective_scan
%cd selective_scan
!pip install .
%cd ..

# Step 5: Verify setup
import torch
print(f"✅ PyTorch {torch.__version__}")
print(f"✅ CUDA: {torch.cuda.is_available()}")
print(f"✅ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Step 6: Upload dataset (do this manually or via Drive)
# Ensure dataset is at: data/Cirrhosis_T2_3D/

# Step 7: Train
!python train.py
```

---

## 🎯 Colab Advantages

1. **Free GPU** - T4 GPU for training (much faster than CPU)
2. **No setup issues** - Python 3.9+ and CUDA pre-configured
3. **Easy sharing** - Share notebook with colleagues
4. **Persistent Drive** - Save results to Drive
5. **Automatic updates** - Packages stay current

---

## 📦 Dataset Upload Options

### Method 1: Google Drive (Recommended)
```python
# 1. Upload dataset to Google Drive
# 2. Mount and copy:
from google.colab import drive
drive.mount('/content/drive')
!cp -r "/content/drive/MyDrive/datasets/Cirrhosis_T2_3D" "./data/"
```

### Method 2: Direct Upload
```python
# Use Colab file browser:
# Files → Upload to session storage
# Or use Python:
from google.colab import files
uploaded = files.upload()  # Select dataset files
```

### Method 3: Download from URL
```python
# If dataset is available via direct download
!wget -O dataset.zip "dataset-url"
!unzip dataset.zip
!mkdir -p data
!mv extracted_folder data/Cirrhosis_T2_3D
```

---

## ⚡ Quick Commands for Colab

```python
# Check GPU
!nvidia-smi

# List files
!ls -la

# Check dataset
!ls data/Cirrhosis_T2_3D/

# Monitor training (in separate terminal)
# Colab shows output in real-time

# Save results to Drive
!cp -r results /content/drive/MyDrive/saved_results/
```

---

## 💾 Saving Checkpoints in Colab

**Option 1: Save to Drive (Persistent)**
```python
# Modify train.py to save to Drive
save_path = "/content/drive/MyDrive/saved_models/SRMAMamba/"
```

**Option 2: Download Checkpoints**
```python
from google.colab import files
files.download('results/files_T2/SRMAMamba/checkpoint.pth')
```

**Option 3: Use Colab File Browser**
- Right-click file → Download

---

## 🐛 Troubleshooting in Colab

### Issue: Session disconnects
**Solution:** Colab has time limits. Use Drive to save checkpoints frequently.

### Issue: Out of memory
**Solution:** 
- Reduce batch size in train.py
- Use gradient accumulation
- Restart runtime and clear variables

### Issue: Files lost after disconnect
**Solution:** Always save important files to Drive:
```python
!cp -r results /content/drive/MyDrive/backup/
```

---

## 📊 Recommended Colab Workflow

1. **Setup (One-time):**
   - Upload project files to Drive
   - Create setup notebook
   - Install packages

2. **Training (Each session):**
   - Mount Drive
   - Load project
   - Start training
   - Monitor progress
   - Save checkpoints to Drive

3. **Testing:**
   - Load trained checkpoint from Drive
   - Run test.py
   - Download results

---

## 🎯 Next Steps

1. **Create Colab notebook** (use template above)
2. **Upload project files** to Drive or Colab
3. **Install packages** (run pip install commands)
4. **Upload dataset** to `data/Cirrhosis_T2_3D/`
5. **Train model** (`!python train.py`)
6. **Download results** when done

---

**Pro Tip:** Create a template notebook and save it to Drive so you can reuse it!

**Ready to start?** Copy the setup code above into a new Colab notebook! 🚀

