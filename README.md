# 🩺 Explainable Medical Diagnosis using Deep Learning, SHAP & LIME

This project builds a **Deep Learning model** (CNN-based) for **medical image diagnosis** — specifically detecting **Pneumonia from Chest X-rays** — and integrates **Explainable AI (XAI)** techniques using **SHAP** and **LIME** to visualize *why* the model made its decision.  

It also includes a **frontend** and **backend** for real-time predictions and explanations via a web interface.

---

## 📘 Project Overview

### 🎯 Goal
- Detect **Pneumonia** from chest X-ray images.
- Provide **visual explanations** using SHAP and LIME.
- Enable **user-friendly web interface** for uploading X-rays and viewing results.

### 🧩 Features
- CNN model (ResNet18 or custom CNN)
- SHAP and LIME-based explainability
- Web App: Flask backend + HTML/CSS/JS frontend
- Visualization of important image regions
- Comparative analysis of SHAP vs LIME

---

## 🧠 Technologies Used

| Category | Tools / Frameworks |
|-----------|--------------------|
| Programming | Python |
| Deep Learning | PyTorch |
| Explainable AI | SHAP, LIME |
| Web Backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| Visualization | Matplotlib |
| Dataset | Chest X-Ray Images (Pneumonia) |

---

## 📂 Project Structure

│
├── README.md
├── requirements.txt
├── data/
├── src/
├── app.py
├── frontend_app.py
├── results/


---

## 🗃️ Dataset

**Dataset:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

**Structure:**
data/chest_xray/
├── train/
│ ├── NORMAL/
│ └── PNEUMONIA/
├── val/
└── test/


---

## ⚙️ Installation

### 1️⃣ Clone the project
```bash
git clone https://github.com/yourusername/Explainable_Medical_Diagnosis.git
cd Explainable_Medical_Diagnosis

2️⃣ Install dependencies
pip install -r requirements.txt

🚀 Usage
🔹 Step 1: Train the model
python -m src.train

🔹 Step 2: Generate explanations
python -m src.explain

🔹 Step 3: Run the web app
python backend/app.py


Open browser:

http://127.0.0.1:5000


Upload an X-ray image and view:

Model prediction (Normal / Pneumonia)

SHAP & LIME heatmaps


---

If you want, I can **generate a clean PNG diagram** of this workflow that you can include in the README instead of the ASCII version.  

Do you want me to do that next?
