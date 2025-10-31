"""
Model Testing Script - Tests SRMA-Mamba model performance
Run this to evaluate if your trained model is performing well
"""

import os
import sys
import torch
import numpy as np
from glob import glob
from tqdm import tqdm

print("=" * 70)
print("SRMA-Mamba Model Performance Testing")
print("=" * 70)

# Check for checkpoint
print("\n📁 Step 1: Checking for trained model checkpoint...")
modality = 'T2'  # Change to 'T1' if needed
save_path = os.path.join("results", f"files_{modality}", "SRMAMamba")
checkpoint_path = os.path.join(save_path, "checkpoint.pth")

if os.path.exists(checkpoint_path):
    print(f"   ✅ Checkpoint found: {checkpoint_path}")
    checkpoint_size = os.path.getsize(checkpoint_path) / (1024**2)
    print(f"   📦 Checkpoint size: {checkpoint_size:.2f} MB")
    test_init = False
else:
    print(f"   ❌ Checkpoint NOT found at: {checkpoint_path}")
    print(f"   📝 You need to train the model first using: python train.py")
    print(f"   OR if you have a checkpoint elsewhere, update checkpoint_path in this script")
    
    # Ask if user wants to test model initialization anyway
    print("\n   Will test model initialization (without trained weights)")
    print("   This will verify the model architecture works but won't show performance.")
    test_init = True

# Load model
print("\n🔧 Step 2: Loading model...")
try:
    from configs.model_configs import build_SRMAMamba
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    
    model = build_SRMAMamba()
    model = model.to(device)
    print(f"   ✅ Model architecture loaded")
    
    # Load checkpoint if available
    if os.path.exists(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.eval()
            print(f"   ✅ Checkpoint loaded successfully")
        except Exception as e:
            print(f"   ⚠️  Warning: Could not load checkpoint: {e}")
            print(f"   Testing with random weights instead...")
    else:
        model.eval()
        print(f"   ⚠️  Testing with random (untrained) weights")
        
except Exception as e:
    print(f"   ❌ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check dataset
print("\n📊 Step 3: Checking test dataset...")
dataset_path = f"data/Cirrhosis_{modality}_3D"

if os.path.exists(dataset_path):
    test_x = sorted(glob(os.path.join(dataset_path, "test_images", "*.nii.gz")))
    test_y = sorted(glob(os.path.join(dataset_path, "test_masks", "*.nii.gz")))
    
    print(f"   ✅ Dataset found: {dataset_path}")
    print(f"   📁 Test images: {len(test_x)}")
    print(f"   📁 Test masks: {len(test_y)}")
    
    if len(test_x) == 0:
        print(f"   ⚠️  No test images found!")
        print(f"   📝 Testing with dummy data instead...")
        test_with_dummy = True
    elif len(test_x) != len(test_y):
        print(f"   ⚠️  Mismatch: {len(test_x)} images vs {len(test_y)} masks")
        test_with_dummy = True
    else:
        test_with_dummy = False
else:
    print(f"   ❌ Dataset not found at: {dataset_path}")
    print(f"   📝 Testing with dummy data instead...")
    test_with_dummy = True

# Test forward pass with dummy data
print("\n🧪 Step 4: Testing model forward pass...")
try:
    dummy_input = torch.randn(1, 1, 224, 224, 64).to(device)
    
    with torch.no_grad():
        out1, out2, out3, out4 = model(dummy_input)
        
    print(f"   ✅ Forward pass successful!")
    print(f"   📐 Input shape: {dummy_input.shape}")
    print(f"   📐 Output shapes:")
    print(f"      - Output 1: {out1.shape}")
    print(f"      - Output 2: {out2.shape}")
    print(f"      - Output 3: {out3.shape}")
    print(f"      - Output 4: {out4.shape}")
    
    # Check for NaN/Inf
    all_finite = all(
        torch.isfinite(out).all() 
        for out in [out1, out2, out3, out4]
    )
    
    if all_finite:
        print(f"   ✅ All outputs are finite (no NaN/Inf)")
    else:
        print(f"   ⚠️  Warning: Some outputs contain NaN/Inf!")
        
    # Check output range
    print(f"   📊 Output value ranges:")
    print(f"      - Output 1: [{out1.min().item():.3f}, {out1.max().item():.3f}]")
    print(f"      - Output 2: [{out2.min().item():.3f}, {out2.max().item():.3f}]")
    print(f"      - Output 3: [{out3.min().item():.3f}, {out3.max().item():.3f}]")
    print(f"      - Output 4: [{out4.min().item():.3f}, {out4.max().item():.3f}]")
    
except Exception as e:
    print(f"   ❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test with actual data if available
if not test_with_dummy and len(test_x) > 0:
    print("\n📈 Step 5: Testing on actual test data...")
    
    try:
        from monai import transforms
        from monai.inferers import SlidingWindowInferer
        from utils import calculate_metrics2
        from torch.cuda.amp import autocast
        
        # Setup transforms
        test_transform = transforms.Compose([
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstD(keys=["image", "label"], channel_dim="no_channel"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image", "label"])
        ])
        
        # Setup sliding window inference
        size = [224, 224, 64]
        window_infer = SlidingWindowInferer(roi_size=size, sw_batch_size=1, overlap=0.25)
        
        # Test on first few samples
        num_test_samples = min(5, len(test_x))
        print(f"   Testing on {num_test_samples} samples...")
        
        metrics_score = [0.0] * 8
        time_taken = []
        
        for i in tqdm(range(num_test_samples), desc="   Processing"):
            try:
                # Load data
                data = {"image": test_x[i], "label": test_y[i]}
                augmented = test_transform(data)
                image_data = augmented["image"]
                label_data = augmented["label"]
                
                image_data = image_data.unsqueeze(0).to(device, dtype=torch.float32)
                label_data = label_data.unsqueeze(0).to(device, dtype=torch.float32)
                
                # Inference
                import time
                start_time = time.time()
                
                with torch.no_grad():
                    with autocast():
                        y1, y2, y3, y4 = window_infer(image_data, model)
                        pred = y1
                        
                        pred = torch.sigmoid(pred)
                        pred = pred[0]
                
                end_time = time.time() - start_time
                time_taken.append(end_time)
                
                # Calculate metrics
                score = calculate_metrics2(label_data, pred)
                metrics_score = [m + s for m, s in zip(metrics_score, score)]
                
            except Exception as e:
                print(f"\n   ⚠️  Error processing sample {i}: {e}")
                continue
        
        # Calculate averages
        if len(time_taken) > 0:
            mean_time = np.mean(time_taken)
            mean_fps = 1 / mean_time if mean_time > 0 else 0
            
            metrics_avg = [m / num_test_samples for m in metrics_score]
            
            print(f"\n   📊 Performance Metrics (on {num_test_samples} samples):")
            print(f"   " + "-" * 60)
            print(f"   Jaccard Index:    {metrics_avg[0]:.4f}")
            print(f"   Dice Score (F1):  {metrics_avg[1]:.4f}")
            print(f"   Recall:           {metrics_avg[2]:.4f}")
            print(f"   Precision:        {metrics_avg[3]:.4f}")
            print(f"   Accuracy:         {metrics_avg[4]:.4f}")
            print(f"   F2 Score:         {metrics_avg[5]:.4f}")
            print(f"   HD95 (mm):        {metrics_avg[7]:.2f}")
            print(f"   ASSD (mm):        {metrics_avg[6]:.2f}")
            print(f"   " + "-" * 60)
            print(f"   Inference Speed: {mean_fps:.2f} FPS")
            print(f"   Avg Time/Sample:  {mean_time:.2f} seconds")
            
            # Performance assessment
            print(f"\n   🎯 Performance Assessment:")
            dice_score = metrics_avg[1]
            jaccard = metrics_avg[0]
            hd95 = metrics_avg[7]
            
            if dice_score > 0.90 and jaccard > 0.85 and hd95 < 5.0:
                print(f"   ✅ EXCELLENT! Model is performing very well!")
                print(f"      - Dice > 0.90, Jaccard > 0.85, HD95 < 5mm")
            elif dice_score > 0.80 and jaccard > 0.70 and hd95 < 10.0:
                print(f"   ✅ GOOD! Model is performing well.")
                print(f"      - Dice > 0.80, Jaccard > 0.70, HD95 < 10mm")
            elif dice_score > 0.70:
                print(f"   ⚠️  MODERATE. Model performance could be improved.")
                print(f"      - Consider training longer or tuning hyperparameters")
            else:
                print(f"   ❌ POOR. Model needs significant improvement.")
                print(f"      - Check training process, data quality, or hyperparameters")
                if not os.path.exists(checkpoint_path):
                    print(f"      - NOTE: You're using random weights! Train the model first.")
        
    except Exception as e:
        print(f"   ⚠️  Could not test on real data: {e}")
        print(f"   Model architecture test passed, but need real data for performance metrics")
        import traceback
        traceback.print_exc()

else:
    print("\n⚠️  Step 5: Skipped (no test data available)")
    print(f"   To test performance, you need:")
    print(f"   1. Trained checkpoint: {checkpoint_path}")
    print(f"   2. Test dataset at: {dataset_path}")

# Summary
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)

if os.path.exists(checkpoint_path):
    print("✅ Model checkpoint found and loaded")
else:
    print("⚠️  No trained checkpoint found - using random weights")

if not test_with_dummy and len(test_x) > 0:
    print("✅ Test data available and processed")
else:
    print("⚠️  No test data available - only architecture tested")

print("\n📝 Recommendations:")
if not os.path.exists(checkpoint_path):
    print("   1. Train the model: python train.py")
    print("   2. Wait for checkpoint to be saved")
    print("   3. Re-run this test script")
    
if test_with_dummy or len(test_x) == 0:
    print("   1. Download dataset from: https://osf.io/cuk24/files/osfstorage")
    print("   2. Organize in: data/Cirrhosis_T2_3D/")
    print("   3. Re-run this test script")

print("\n" + "=" * 70)

