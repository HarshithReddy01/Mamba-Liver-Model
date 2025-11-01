"""
Google Colab Setup Script for SRMA-Mamba
Run this in Google Colab cells
"""

# ============================================
# Cell 1: Install Dependencies
# ============================================
print("📦 Installing packages...")

# Install PyTorch with CUDA 11.8
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install core packages
!pip install monai pyyaml yacs nibabel opencv-python pandas tqdm scikit-learn

# Install ML/AI packages
!pip install huggingface-hub tokenizers transformers timm fvcore

# Install mamba-ssm (should work in Colab!)
!pip install mamba-ssm

print("✅ Packages installed!")

# ============================================
# Cell 2: Mount Google Drive (if needed)
# ============================================
from google.colab import drive
drive.mount('/content/drive')

# Navigate to your project (update path)
%cd /content/drive/MyDrive/your-path/Liver\ Cirr\ Model

# OR if uploaded directly to Colab:
# %cd /content/your-project-folder

# ============================================
# Cell 3: Build selective_scan
# ============================================
%cd selective_scan
!pip install .
%cd ..

print("✅ selective_scan built!")

# ============================================
# Cell 4: Verify Setup
# ============================================
import torch
import sys

print("=" * 60)
print("SETUP VERIFICATION")
print("=" * 60)
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️  No GPU detected - training will be slow")

# Test imports
try:
    import monai
    print(f"✅ MONAI: {monai.__version__}")
except:
    print("❌ MONAI not installed")

try:
    import mamba_ssm
    print("✅ mamba-ssm installed")
except:
    print("❌ mamba-ssm not installed")

try:
    from configs.model_configs import build_SRMAMamba
    model = build_SRMAMamba()
    print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
except Exception as e:
    print(f"❌ Model loading failed: {e}")

print("=" * 60)

# ============================================
# Cell 5: Check Dataset
# ============================================
import os
from glob import glob

dataset_path = 'data/Cirrhosis_T2_3D'

if os.path.exists(dataset_path):
    train_x = glob(os.path.join(dataset_path, 'train_images', '*.nii.gz'))
    train_y = glob(os.path.join(dataset_path, 'train_masks', '*.nii.gz'))
    test_x = glob(os.path.join(dataset_path, 'test_images', '*.nii.gz'))
    test_y = glob(os.path.join(dataset_path, 'test_masks', '*.nii.gz'))
    
    print(f"✅ Dataset found!")
    print(f"   Train: {len(train_x)} images, {len(train_y)} masks")
    print(f"   Test:  {len(test_x)} images, {len(test_y)} masks")
else:
    print(f"❌ Dataset not found at: {dataset_path}")
    print(f"📝 Upload dataset to: {dataset_path}/")

# ============================================
# Cell 6: Train Model
# ============================================
# Uncomment when ready to train:
# !python train.py

# ============================================
# Cell 7: Test Model (after training)
# ============================================
# Uncomment after training:
# !python test_model.py

# ============================================
# Cell 8: Download Results
# ============================================
from google.colab import files

# Download checkpoint
# files.download('results/files_T2/SRMAMamba/checkpoint.pth')

# Or download entire results folder
# !zip -r results.zip results/
# files.download('results.zip')

print("Setup complete! Ready to train.")
print("Next: Upload dataset and run train.py")