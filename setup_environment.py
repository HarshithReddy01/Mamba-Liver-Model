"""
Environment Setup Script for SRMA-Mamba
This script helps verify and guide environment setup
"""

import os
import sys
import subprocess

print("=" * 70)
print("SRMA-Mamba Environment Setup Assistant")
print("=" * 70)

def run_command(cmd, check=True):
    """Run a shell command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if check and result.returncode != 0:
            return False, result.stderr
        return True, result.stdout
    except Exception as e:
        return False, str(e)

def check_python_version():
    """Check if Python version is compatible"""
    print("\n1. Checking Python version...")
    version = sys.version_info
    print(f"   Current Python: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor == 9:
        print("   ✅ Python 3.9 detected (recommended)")
        return True
    elif version.major == 3 and version.minor >= 8:
        print(f"   ⚠️  Python {version.major}.{version.minor} (should work, but 3.9 is recommended)")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor} may not be compatible")
        print("   📝 Recommended: Python 3.9.0")
        return False

def check_conda():
    """Check if conda is installed"""
    print("\n2. Checking Conda installation...")
    success, output = run_command("conda --version", check=False)
    if success:
        print(f"   ✅ Conda found: {output.strip()}")
        return True
    else:
        print("   ❌ Conda not found")
        print("   📝 Install Miniconda or Anaconda from: https://docs.conda.io/en/latest/miniconda.html")
        return False

def check_cuda():
    """Check if CUDA is available"""
    print("\n3. Checking CUDA installation...")
    
    # Check nvidia-smi
    success, output = run_command("nvidia-smi", check=False)
    if success:
        lines = output.split('\n')
        for line in lines:
            if 'CUDA Version' in line:
                cuda_version = line.split('CUDA Version:')[1].split()[0]
                print(f"   ✅ NVIDIA GPU detected")
                print(f"   ✅ CUDA Version: {cuda_version}")
                return True
    
    print("   ⚠️  NVIDIA GPU/CUDA not detected")
    print("   📝 You can still set up environment, but GPU training won't work")
    return False

def create_conda_env():
    """Create conda environment"""
    print("\n4. Creating Conda environment...")
    env_name = "SRMA-Mamba"
    
    # Check if environment exists
    success, output = run_command(f"conda env list", check=False)
    if success and env_name in output:
        print(f"   ✅ Environment '{env_name}' already exists")
        print(f"   📝 Activate it with: conda activate {env_name}")
        return True
    
    # Create environment
    print(f"   📝 Creating environment '{env_name}' with Python 3.9...")
    success, output = run_command(f"conda create -n {env_name} python=3.9 -y", check=False)
    if success:
        print(f"   ✅ Environment '{env_name}' created successfully")
        print(f"   📝 Activate it with: conda activate {env_name}")
        return True
    else:
        print(f"   ❌ Failed to create environment: {output}")
        return False

def install_pytorch():
    """Install PyTorch with CUDA support"""
    print("\n5. Installing PyTorch...")
    
    # Check if CUDA is available
    cuda_available = check_cuda()
    
    if cuda_available:
        print("   📝 Installing PyTorch with CUDA 11.8 support...")
        cmd = "pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118"
    else:
        print("   📝 Installing PyTorch (CPU version)...")
        cmd = "pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2"
    
    success, output = run_command(cmd, check=False)
    if success:
        print("   ✅ PyTorch installed successfully")
        
        # Verify installation
        verify_cmd = "python -c \"import torch; print('CUDA available:', torch.cuda.is_available())\""
        success, output = run_command(verify_cmd, check=False)
        if success:
            print(f"   {output.strip()}")
        return True
    else:
        print(f"   ❌ Failed to install PyTorch: {output}")
        return False

def install_requirements():
    """Install requirements from requirements.txt"""
    print("\n6. Installing project requirements...")
    
    if not os.path.exists("requirements.txt"):
        print("   ❌ requirements.txt not found")
        return False
    
    success, output = run_command("pip install -r requirements.txt", check=False)
    if success:
        print("   ✅ Requirements installed")
        return True
    else:
        print(f"   ⚠️  Some packages may have failed: {output}")
        return False

def install_selective_scan():
    """Install selective_scan package"""
    print("\n7. Installing selective_scan (CUDA kernels)...")
    
    if not os.path.exists("selective_scan"):
        print("   ❌ selective_scan directory not found")
        return False
    
    original_dir = os.getcwd()
    try:
        os.chdir("selective_scan")
        success, output = run_command("pip install .", check=False)
        os.chdir(original_dir)
        
        if success:
            print("   ✅ selective_scan installed")
            return True
        else:
            print(f"   ⚠️  Installation may have issues: {output}")
            return False
    except Exception as e:
        os.chdir(original_dir)
        print(f"   ❌ Error: {e}")
        return False

def install_triton():
    """Install Triton"""
    print("\n8. Installing Triton...")
    success, output = run_command("pip install triton==2.2.0", check=False)
    if success:
        print("   ✅ Triton installed")
        return True
    else:
        print(f"   ⚠️  Triton installation may have issues: {output}")
        print("   📝 This is optional but recommended for performance")
        return False

def verify_installation():
    """Verify all packages are installed"""
    print("\n9. Verifying installation...")
    
    packages = {
        'torch': 'PyTorch',
        'torchvision': 'Torchvision',
        'monai': 'MONAI',
        'mamba_ssm': 'Mamba SSM',
        'nibabel': 'Nibabel',
        'numpy': 'NumPy',
        'tqdm': 'TQDM'
    }
    
    all_ok = True
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} - NOT installed")
            all_ok = False
    
    return all_ok

def main():
    """Main setup function"""
    print("\n" + "=" * 70)
    print("SETUP PROCESS")
    print("=" * 70)
    
    steps_completed = 0
    total_steps = 9
    
    # Step 1: Python version
    if check_python_version():
        steps_completed += 1
    
    # Step 2: Conda
    if check_conda():
        steps_completed += 1
        print("\n   Would you like to create/use conda environment? (Recommended)")
        print("   Run manually: conda create -n SRMA-Mamba python=3.9 -y")
    else:
        print("\n   ⚠️  Conda not found - will use system Python/pip")
    
    # Step 3: CUDA check
    if check_cuda():
        steps_completed += 1
    
    print("\n" + "=" * 70)
    print("AUTOMATED SETUP OPTIONS")
    print("=" * 70)
    print("\nChoose an option:")
    print("1. Create conda environment only")
    print("2. Install all packages (assumes conda env is activated)")
    print("3. Verify installation only")
    print("4. Full setup (create env + install everything)")
    
    choice = input("\nEnter choice (1-4) or 'q' to quit: ").strip()
    
    if choice == '1':
        if create_conda_env():
            print("\n✅ Environment created! Next steps:")
            print("   conda activate SRMA-Mamba")
            print("   python setup_environment.py")
            print("   (Choose option 2 to install packages)")
    
    elif choice == '2':
        print("\n📦 Installing packages...")
        if install_pytorch():
            steps_completed += 1
        if install_requirements():
            steps_completed += 1
        if install_selective_scan():
            steps_completed += 1
        if install_triton():
            steps_completed += 1
        if verify_installation():
            steps_completed += 1
            
        print(f"\n✅ Installed {steps_completed}/5 package groups")
    
    elif choice == '3':
        if verify_installation():
            print("\n✅ All packages verified!")
        else:
            print("\n❌ Some packages missing - run option 2 to install")
    
    elif choice == '4':
        print("\n🚀 Full setup starting...")
        if create_conda_env():
            print("\n⚠️  Please activate the environment first:")
            print("   conda activate SRMA-Mamba")
            print("   Then run this script again and choose option 2")
        else:
            print("\n⚠️  Could not create environment - check conda installation")
    
    elif choice.lower() == 'q':
        print("\n👋 Setup cancelled")
        return
    
    else:
        print("\n❌ Invalid choice")
        return
    
    print("\n" + "=" * 70)
    print("SETUP COMPLETE")
    print("=" * 70)
    
    print("\n📝 Next steps:")
    print("   1. Verify setup: python quick_start.py")
    print("   2. Check dataset: data/Cirrhosis_T2_3D/")
    print("   3. Train model: python train.py")
    print("   4. Test model: python test_model.py")

if __name__ == "__main__":
    main()

