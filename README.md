# 🩸 Retinal Blood Vessel Segmentation using U-Net  
*Deep Learning–based Semantic Segmentation for Medical Imaging Applications*

[![Python](https://img.shields.io/badge/Python-3.12-blue)]()
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)]()
[![U-Net](https://img.shields.io/badge/Model-U--Net-green)]()
[![Dataset](https://img.shields.io/badge/Dataset-ICPR--Retinal--Vessels-red)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey)]()

---

## 🧾 Abstract
This project implements **semantic segmentation** of retinal blood vessels from fundus images using a **U-Net convolutional neural network**.  
The model was trained on the **ICPR Retinal Blood Vessel Dataset** and achieved a **mean Intersection over Union (mIoU)** of **71.48%** on the test set.  
Accurate segmentation of retinal vessels is essential for early detection of diseases like **diabetic retinopathy**, **hypertension**, and **glaucoma**.  

---

## 🩺 Problem Statement
Manual segmentation of retinal blood vessels is time-consuming and prone to inconsistency across experts.  
This project automates the process using a **U-Net deep learning architecture** that enables precise segmentation of retinal vessels in high-resolution fundus images.

---

## ⚙️ Technical Overview

| Component | Specification |
|------------|----------------|
| **Architecture** | U-Net |
| **Framework** | PyTorch |
| **Language** | Python |
| **Dataset** | ICPR Retinal Blood Vessel Dataset |
| **Loss Function** | CrossEntropyLoss |
| **Optimizer** | Adam (lr=1e-4) |
| **Evaluation Metric** | Mean IoU (mIoU) |
| **Final mIoU** | **71.48%** |

---

## 🧠 Methodology

### 1. Dataset & Preprocessing
- Dataset contains **268 train** and **112 test** images in `.tif` format.  
- Two classes: *vessel* and *background*.  
- Implemented a **custom PyTorch Dataset class** for efficient image–mask pairing.  
- Images resized to **512×512** and normalized.  

### 2. Model Architecture
- **U-Net** with encoder–decoder structure and skip connections.  
- Encoder: multiple convolutional blocks (Conv → BatchNorm → ReLU) with MaxPooling.  
- Decoder: Upsampling via bilinear interpolation and concatenation with encoder feature maps.  
- Output layer: 2-channel segmentation map representing vessel and background.  

### 3. Training Details
- **Epochs:** 30  
- **Batch size:** 8  
- **Optimizer:** Adam (lr = 1e-4)  
- **Loss:** CrossEntropyLoss  
- **Hardware:** NVIDIA GPU (CUDA)  
- **Metrics:** Train/Validation Loss, Accuracy, and Mean IoU (mIoU)  

---

## 📊 Results

| Metric | Train | Validation |
|--------|--------|-------------|
| **Loss** | ↓ 0.57 → 0.15 | ↓ 0.65 → 0.16 |
| **mIoU** | ↑ 0.47 → 0.74 | ↑ 0.45 → 0.71 |
| **Final Test mIoU** | **71.48%** |

The model successfully captures fine vessel structures, even in low-contrast and peripheral regions of the retina.

---

## 🎯 Inference Visualization
Example outputs showing:
- **Left:** Original Fundus Image  
- **Center:** Ground Truth Vessel Mask  
- **Right:** Model Prediction Overlay  

These qualitative results demonstrate high visual fidelity and accurate vessel delineation.

---

## ⚡ Setup & Usage

### 1. Clone Repository
```bash
git clone https://github.com/<your-username>/Retinal-Vessel-Segmentation-UNet.git
cd Retinal-Vessel-Segmentation-UNet
```
### 2. Create Virtual Environment (Optional but Recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Dataset Preparation

Download the ICPR Retinal Blood Vessel Dataset and structure it as follows:

```bash
icpr_prepared/
├── train_images/
├── train_labels/
├── test_images/
└── test_labels/
```

### 5. Train the Model
```bash
python main.ipynb
```

(or open in Jupyter Notebook and run all cells)


### 6. Run Inference

To evaluate on test data:
```bash
python inference.py --weights vessel_segmentation_model.pt --data icpr_prepared/
```

Predicted masks and overlayed visualizations will be saved in the results/ directory.

🧰 Repository Structure

```bibtex

├── icpr_prepared/
│   ├── train_images/
│   ├── train_labels/
│   ├── test_images/
│   └── test_labels/
├── main.ipynb
├── requirements.txt
└── README.md

```
## 🧾 Citation

```bibtex
@project{RetinalVesselSegmentation2025,
  title   = {Retinal Blood Vessel Segmentation using U-Net},
  author  = {Ahmad Ishaque Karimi},
  year    = {2025},
  note    = {GitHub repository: https://github.com/Ahmadishaque/Retinal_Vessel_Segmentation_UNet}
}
```

## 🚀 Future Work
- Integrate Attention U-Net or UNet++ for better feature propagation.
- Apply Dice loss or Tversky loss to handle class imbalance.
- Explore CLAHE preprocessing for vessel enhancement.
- Deploy lightweight U-Net variant for edge inference on medical devices.

---

## 👨‍💻 Author

**Ahmad Ishaque Karimi**  
Graduate Student — Data Science & Computer Vision Research  
📧 ahmadishaquekarimi@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/ahmadishaquekarimi/)

---
