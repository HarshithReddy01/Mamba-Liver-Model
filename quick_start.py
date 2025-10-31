"""
Quick Start Verification Script for SRMA-Mamba
Run this to verify your setup is correct before training
"""

import os
import sys
import torch
from glob import glob

print("=" * 60)
print("SRMA-Mamba Quick Start Verification")
print("=" * 60)

# 1. Check Python version
print("\n1. Checking Python version...")
python_version = sys.version_info
if python_version.major == 3 and python_version.minor == 9:
    print(f"   ✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
else:
    print(f"   ⚠️  Python {python_version.major}.{python_version.minor} (Recommended: 3.9.0)")

# 2. Check CUDA
print("\n2. Checking CUDA...")
try:
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device_count = torch.cuda.device_count()
        cuda_version = torch.version.cuda
        print(f"   ✅ CUDA available: {device_name}")
        print(f"   ✅ CUDA devices: {device_count}")
        print(f"   ✅ CUDA version: {cuda_version}")
        print(f"   ✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("   ❌ CUDA not available - GPU training will not work!")
        print("   ⚠️  You can still test model initialization on CPU")
except Exception as e:
    print(f"   ❌ CUDA check failed: {e}")

# 3. Check required packages
print("\n3. Checking required packages...")
required_packages = {
    'torch': 'PyTorch',
    'torchvision': 'Torchvision',
    'monai': 'MONAI',
    'mamba_ssm': 'Mamba SSM',
    'nibabel': 'Nibabel',
    'numpy': 'NumPy',
    'tqdm': 'TQDM'
}

missing_packages = []
for package, name in required_packages.items():
    try:
        __import__(package)
        print(f"   ✅ {name} installed")
    except ImportError:
        print(f"   ❌ {name} NOT installed")
        missing_packages.append(package)

if missing_packages:
    print(f"\n   Install missing packages: pip install {' '.join(missing_packages)}")

# 4. Check model files
print("\n4. Checking model files...")
model_files = [
    'model/SRMAMamba.py',
    'model/vmamba2.py',
    'configs/model_configs.py',
    'configs/vssm1/vmambav2_tiny_224.yaml'
]

for file_path in model_files:
    if os.path.exists(file_path):
        print(f"   ✅ {file_path}")
    else:
        print(f"   ❌ {file_path} - MISSING!")

# 5. Check dataset
print("\n5. Checking dataset...")
dataset_path = 'data/Cirrhosis_T2_3D'
if os.path.exists(dataset_path):
    train_x = glob(os.path.join(dataset_path, 'train_images', '*.nii.gz'))
    train_y = glob(os.path.join(dataset_path, 'train_masks', '*.nii.gz'))
    valid_x = glob(os.path.join(dataset_path, 'valid_images', '*.nii.gz'))
    valid_y = glob(os.path.join(dataset_path, 'valid_masks', '*.nii.gz'))
    test_x = glob(os.path.join(dataset_path, 'test_images', '*.nii.gz'))
    test_y = glob(os.path.join(dataset_path, 'test_masks', '*.nii.gz'))
    
    print(f"   ✅ Dataset found at: {dataset_path}")
    print(f"   ✅ Train: {len(train_x)} images, {len(train_y)} masks")
    print(f"   ✅ Valid: {len(valid_x)} images, {len(valid_y)} masks")
    print(f"   ✅ Test:  {len(test_x)} images, {len(test_y)} masks")
    
    if len(train_x) == 0:
        print("   ⚠️  No training images found - dataset may be empty!")
else:
    print(f"   ❌ Dataset not found at: {dataset_path}")
    print(f"   📝 Please download from: https://osf.io/cuk24/files/osfstorage")
    print(f"   📝 Organize as: data/Cirrhosis_T2_3D/train_images/, train_masks/, etc.")

# 6. Test model initialization
print("\n6. Testing model initialization...")
try:
    from configs.model_configs import build_SRMAMamba
    
    model = build_SRMAMamba()
    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   ✅ Model loaded successfully!")
    print(f"   ✅ Total parameters: {param_count/1e6:.2f}M")
    print(f"   ✅ Trainable parameters: {trainable_count/1e6:.2f}M")
except Exception as e:
    print(f"   ❌ Model initialization failed: {e}")
    print(f"   📝 Check configs/model_configs.py and model/SRMAMamba.py")

# 7. Test forward pass
print("\n7. Testing forward pass...")
try:
    if 'model' in locals():
        model.eval()
        dummy_input = torch.randn(1, 1, 224, 224, 64)
        
        if torch.cuda.is_available():
            model = model.cuda()
            dummy_input = dummy_input.cuda()
            print(f"   ℹ️  Using GPU for test")
        else:
            print(f"   ⚠️  Using CPU for test (will be slow)")
        
        with torch.no_grad():
            out1, out2, out3, out4 = model(dummy_input)
            
            print(f"   ✅ Forward pass successful!")
            print(f"   ✅ Output shapes:")
            print(f"      - out1: {out1.shape}")
            print(f"      - out2: {out2.shape}")
            print(f"      - out3: {out3.shape}")
            print(f"      - out4: {out4.shape}")
            
            # Verify outputs are finite
            all_finite = all(
                torch.isfinite(out).all() 
                for out in [out1, out2, out3, out4]
            )
            if all_finite:
                print(f"   ✅ All outputs are finite (no NaN/Inf)")
            else:
                print(f"   ⚠️  Warning: Some outputs contain NaN/Inf")
    else:
        print(f"   ⚠️  Skipped (model not initialized)")
except Exception as e:
    print(f"   ❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 60)
print("VERIFICATION SUMMARY")
print("=" * 60)

issues = []
if not torch.cuda.is_available():
    issues.append("CUDA not available")
if missing_packages:
    issues.append(f"Missing packages: {', '.join(missing_packages)}")
if not os.path.exists(dataset_path):
    issues.append("Dataset not found")

if not issues:
    print("\n✅ All checks passed! You're ready to train.")
    print("\n📝 Next steps:")
    print("   1. Review SETUP_GUIDE.md for detailed instructions")
    print("   2. Run: python train.py")
    print("   3. Monitor: results/files_T2/SRMAMamba/train_log.txt")
else:
    print("\n⚠️  Issues found:")
    for issue in issues:
        print(f"   - {issue}")
    print("\n📝 Please fix these issues before training.")
    print("   See SETUP_GUIDE.md for detailed instructions.")

print("\n" + "=" * 60)

