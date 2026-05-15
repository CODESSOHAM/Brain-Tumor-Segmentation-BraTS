# 3D Brain Tumor Segmentation using 3D U-Net (BraTS 2018)

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-orange.svg)
![MONAI](https://img.shields.io/badge/library-MONAI-green.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

This repository contains a professional implementation of a **3D U-Net** for automated medical image segmentation. Developed by **Soham Aich**, this project focuses on identifying complex tumor structures in volumetric MRI data from the **BraTS 2018** (Brain Tumor Segmentation) dataset.

---

## 🚀 Key Features
* **Volumetric 3D Analysis:** Utilizes 3D convolutions to preserve anatomical spatial context across MRI slices, outperforming standard 2D approaches.
* **Specialized Preprocessing:** Leverages the **MONAI** framework for medical-grade transforms, including intensity scaling, random cropping, and RAS orientation alignment.
* **Multi-Modal Integration:** Processes 4 distinct MRI modalities (T1, T1c, T2, FLAIR) simultaneously as input channels.
* **Optimized Loss Function:** Implements a hybrid **Dice Cross-Entropy Loss** to tackle class imbalance, ensuring accurate segmentation of small tumor regions within the brain volume.

---

## 🛠️ Tech Stack
* **Core Framework:** PyTorch
* **Medical AI Library:** MONAI
* **Imaging Utilities:** Nibabel (NIfTI processing), Scikit-Image
* **Data Science:** NumPy, Matplotlib, tqdm

---

## 📂 Project Structure
```text
├── data/               # Local dataset directory (not uploaded to GitHub)
├── src/
│   ├── model.py        # 3D U-Net Architecture implementation
│   ├── dataset.py      # MONAI transforms and data loading logic
│   └── train.py        # Training loop and optimization
├── notebooks/
│   └── detection-2018.ipynb  # Original research and exploration notebook
├── requirements.txt    # Project dependencies
└── README.md           # Documentation
```
---

## 📊 Dataset & Visuals
The project utilizes the **BraTS 2018** dataset, which consists of multi-institutional pre-operative MRI scans. Each sample includes four MRI modalities: **T1, T1c, T2, and FLAIR**.

The model is designed to segment three nested sub-regions:
1. **Whole Tumor (WT):** The entire tumor extent (including edema).
2. **Tumor Core (TC):** The bulk of the tumor (excluding edema).
3. **Enhancing Tumor (ET):** The active, gadolinium-enhancing part of the tumor.



> **Note:** For a deep dive into the visualization of segmentation masks, prediction overlays, and training loss curves, please refer to the research notebook: `notebooks/detection-2018.ipynb`.

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone [https://github.com/SohamAich/Brain-Tumor-Segmentation.git](https://github.com/SohamAich/Brain-Tumor-Segmentation.git)
cd Brain-Tumor-Segmentation

### 2. Install requirements
It is recommended to use a virtual environment or a Conda environment to keep your dependencies isolated.
```bash
pip install -r requirements.txt
```

### 3. Data Configuration
```text
├── data/
│   ├── train/
│   │   ├── images/  # .nii.gz files (T1, T1ce, T2, FLAIR)
│   │   └── masks/   # .nii.gz files (Segmentation Labels)
```

## 4. Run Training
```text
python src/train.py
```

## 📜 License

This project is licensed under the **MIT License**. This allows for free use, modification, and distribution of the software while ensuring the original author is credited. 

