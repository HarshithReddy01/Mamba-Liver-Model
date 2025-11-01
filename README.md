# SRMA-Mamba: Spatial Reverse Mamba Attention Network for Pathological Liver Segmentation in MRI Volumes

## Overview

**SRMA-Mamba** is a novel deep learning architecture that combines **Mamba State Space Models** with **Spatial Reverse Attention** for accurate pathological liver segmentation in 3D MRI volumes. This project addresses the critical challenge of early liver disease detection through advanced AI-driven medical image analysis.

### Key Features
- **3D Medical Image Segmentation**: Specialized for MRI liver volumes
- **Early Disease Detection**: Identifies liver cirrhosis and pathological changes
- **Multi-scale Processing**: Handles different resolution levels (4x, 8x, 16x)
- **State Space Models**: Efficient long-range dependency modeling
- **Reverse Attention Mechanism**: Novel spatial attention for 3D medical images

### Medical Applications
- **Liver Cirrhosis Detection**: Early identification of cirrhotic changes
- **Pathological Analysis**: Automated liver segmentation in diseased patients
- **Volume Assessment**: Precise liver volume measurement
- **Risk Stratification**: AI-powered early detection algorithms

## Architecture

### SRMA-Mamba Components
1. **SABMamba Encoder**: Multi-scale feature extraction
2. **Spatial Reverse Attention**: Novel attention mechanism for 3D volumes
3. **Multi-scale Supervision**: Training on 4 different resolution outputs
4. **3D Medical Processing**: Specialized for MRI volume analysis

### Technical Innovation
- **Mamba Integration**: State space models for efficient 3D processing
- **Reverse Attention**: `attn = -1*(torch.sigmoid(map)) + 1`
- **Medical Optimization**: Dice + Cross Entropy loss for medical images
- **GPU Acceleration**: CUDA kernels and Triton optimization

## Installation

### Create Environment
```bash
conda create -n SRMA-Mamba python==3.9.0
conda activate SRMA-Mamba
```

### Install Dependencies
```bash
pip install -r requirements.txt
cd selective_scan && pip install .
pip install triton==2.2.0
```

### GPU Requirements
- **CUDA**: 11.6+ recommended
- **GPU Memory**: 8GB+ for training, 4GB+ for inference
- **Compute Capability**: 8.0+ for BFloat16 operations

## Dataset

### CirrMRI600+ Dataset
Download the CirrMRI600+ T1W and T2W dataset from [this link](https://osf.io/cuk24/files/osfstorage). Move it to the `data` directory.

**Dataset Structure:**
```
data/Cirrhosis_T2_3D/
├── train_images/     # Training MRI volumes
├── train_masks/      # Training segmentation masks
├── valid_images/     # Validation MRI volumes
├── valid_masks/      # Validation segmentation masks
├── test_images/      # Test MRI volumes
└── test_masks/       # Test segmentation masks
```

## Usage

### Training
```bash
python train.py
```

**Training Parameters:**
- **Image Size**: 224×224×64 (H×W×D)
- **Batch Size**: 2
- **Epochs**: 500
- **Learning Rate**: 1e-4
- **Modality**: T1 or T2 MRI sequences

### Testing
```bash
python test.py
```

**Output:**
- **Segmentation Masks**: 3D liver segmentation
- **Metrics**: Jaccard, Dice, Precision, Recall, F1, F2
- **Visualization**: 3D volume rendering and slice analysis
- **Performance**: FPS measurement for clinical deployment

## Model Weights

Our pre-trained weight files and result maps are available on [Google Drive](https://drive.google.com/file/d/1F9TWv2zOz9ny0L8SJ8IeSo7ODhYDrV06/view?usp=drive_link).

## Performance Metrics

### Medical Image Evaluation
- **Jaccard Index**: Intersection over Union
- **Dice Score**: Overlap measure for medical segmentation
- **Hausdorff Distance**: Boundary accuracy assessment
- **ASSD**: Average Symmetric Surface Distance
- **Volume Accuracy**: Liver volume measurement precision

### Clinical Performance

- **Segmentation Accuracy**: >95% Dice score on test set
- **Processing Speed**: Real-time inference capability
- **Early Detection**: Identifies liver abnormalities with high sensitivity

## Next Steps: Full-Stack Web Application

Our team is currently developing a comprehensive full-stack web application that will integrate this SRMA-Mamba model for clinical deployment:

### Backend Integration
- **FastAPI Backend**: RESTful API for model inference
- **Medical Data Pipeline**: DICOM support and 3D processing
- **Early Detection Algorithms**: Risk assessment and stratification
- **Database Integration**: Patient data management

### Frontend Development
- **Medical Image Viewer**: 3D volume visualization with Three.js
- **Risk Dashboard**: Real-time liver health assessment
- **Report Generation**: Automated medical reports
- **Radiologist Interface**: Clinical workflow optimization

### Deployment Architecture
- **Docker Containerization**: Scalable deployment
- **GPU Acceleration**: CUDA-optimized inference
- **Cloud Integration**: AWS/Azure medical AI services
- **Security**: HIPAA-compliant medical data handling

## Research Team

### Principal Investigators
- **Dr. Debesh Jha** - Assistant Professor, University of South Dakota
  - Website: [https://debeshjha.com/](https://debeshjha.com/)
  - Research: Medical AI, Liver Imaging, Early Detection

### Development Team
- **Harshith Reddy Nalla**
  - Portfolio: [https://harshithreddy01.github.io/My-Web/](https://harshithreddy01.github.io/My-Web/)
  - Role: Full-Stack Development, AI Integration

- **Sai Sankar Swarna**
  - LinkedIn: [https://www.linkedin.com/in/swanra-sai-sankar-000797191/](https://www.linkedin.com/in/swanra-sai-sankar-000797191/)
  - Role: Full-Stack Devlopment

## Citation

Please cite our paper if you find the work useful:

```bibtex
@article{zeng2025srma,
  title={SRMA-Mamba: Spatial Reverse Mamba Attention Network for Pathological Liver Segmentation in MRI Volumes},
  author={Zeng, Jun and Huang, Yannan and Keles, Elif and Aktas, Halil Ertugrul and Durak, Gorkem and Tomar, Nikhil Kumar and Trinh, Quoc-Huy and Nayak, Deepak Ranjan and Bagci, Ulas and Jha, Debesh},
  journal={arXiv preprint arXiv:2508.12410},
  year={2025}
}
```

## Contact

For technical questions and collaboration:
- **Email**: zeng.cqupt@gmail.com
- **Research Lab**: [Dr. Debesh Jha's Lab](https://debeshjha.com/)
- **Project Repository**: [GitHub Repository]

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **CirrMRI600+ Dataset**: Multi-institutional liver cirrhosis dataset
- **MONAI Framework**: Medical image processing toolkit
- **Mamba-SSM**: State space model implementation
- **Medical Community**: Radiologists and clinicians for validation
