# 🏎️ F1TENTH Autonomous Racing: Recurrent LidarNet



## 📌 Overview

This project introduces **Recurrent LidarNet (RLN)** — a temporal-aware, deep learning architecture for autonomous racing on the F1TENTH platform. RLN leverages sequential 2D LiDAR data using a combination of convolutional, recurrent, and attention mechanisms to enhance spatial-temporal perception and improve driving control.

The repository includes all code, trained models, and datasets required to reproduce results and experiment with the RLN framework.

---

## 📁 Table of Contents

- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Results](#results)
- [Directory Details](#directory-details)
- [Future Work](#future-work)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🧠 Key Features

- **Recurrent LidarNet Architecture**: Combines 1D convolutions, bidirectional LSTM, and attention to extract robust temporal features from LiDAR sequences.
- **End-to-End Learning**: Predicts steering and velocity commands directly from raw LiDAR data.
- **Baseline Comparison**: Includes both RLN and TinyLidarNet (TLN) for benchmarking.
- **Reproducibility**: Contains scripts and configuration files for training, evaluation, and data collection.

---

## ⚙️ Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/f1tenth-recurrent-lidarnet.git
   cd f1tenth-recurrent-lidarnet
   ```

2. **Install dependencies**:
   - Python 3.8+
   - (Recommended) Create a virtual environment
   - Install required packages:
     ```bash
     pip install -r requirements.txt
     ```
   - If `requirements.txt` is missing, manually install:
     ```bash
     pip install tensorflow numpy matplotlib pyyaml
     ```

3. **Prepare the dataset**:
   - Place your training/testing data in the `car_Dataset/` folder.

---

## 🚀 Usage

### 📦 Data Collection

Use the following script to collect driving data from the F1TENTH platform:
```bash
python data_collection.py
```

---

### 🏋️ Training

Train RLN or TLN models with:
```bash
python train.py
```


---

### 🔍 Inference

Evaluate a trained model on test data:
```bash
python inference.py
```
be sure to give the correct path.

---

## 📊 Results

### 📈 Training & Validation Loss

![Loss Curves](Images/Screenshot%202025-05-10%20165238.png)

- RLN achieves lower and more stable validation loss than TLN, indicating improved generalization and reduced overfitting.

---

### 🧱 Network Architecture

![Network Architecture](Images/image.png)

- RLN processes sequences of LiDAR scans using:
  - 1D Convolutional layers
  - Bidirectional LSTM
  - Attention mechanisms
  - Fully connected layers for control outputs

---

## 📂 Directory Details

```
.
├── Figures/           # Report visualizations
├── Images/            # Architecture diagrams & results
├── Models/            # Trained model checkpoints
├── car_Dataset/       # Training & test data
├── data_collection.py # Data collection script
├── inference.py       # Model inference script
├── train_modified.py  # Main training script
├── tln.yml            # Model config file
└── Readme.md          # Project documentation
```




