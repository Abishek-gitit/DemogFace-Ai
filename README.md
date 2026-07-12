# Facial Age & Gender Analyzer

A premium multi-task deep learning application for age estimation, gender classification, and demographic analytics reports using the **UTKFace dataset**. Optimized for Apple Silicon MPS (Metal Performance Shaders) GPU acceleration.

---

## 🚀 Setup Instructions

1. **Activate virtual environment:**
   ```bash
   source .venv/bin/activate
   ```
2. **Install dependencies:** (Already pre-configured during project setup)
   ```bash
   pip install -r requirements.txt
   ```

---

## 📥 Dataset Download

To download and extract the official **Aligned & Cropped Face** images subset (~100MB) of UTKFace:
```bash
python src/download_dataset.py
```
This utility automatically downloads the file and unpacks it into the `data/UTKFace` folder.

*(Note: If Google Drive API limits occur, download the `UTKFace.tar.gz` manually from [official UTKFace site](https://susanqq.github.io/UTKFace/) and extract it into `data/` directory).*

---

## 🏋️ Training the Model

Fine-tune a pre-trained **ResNet-18** model using the UTKFace dataset with MPS GPU acceleration:
```bash
python src/train.py --epochs 10 --batch_size 64
```
* **Backbone Options:** Choose `--backbone resnet18` (default) or `--backbone mobilenet_v3_small` (highly lightweight).
* **Expected Training Time:** ~6 to 8 minutes for 10 epochs on M1 MPS GPU.
* The training script automatically saves the best performing model weights to `models/best_age_gender_resnet18.pth`.

---

## 💻 Running the Web Dashboard

1. **Start the FastAPI Backend server:**
   ```bash
   uvicorn src.app:app --reload --port 8000
   ```
2. **Open the Dashboard:**
   Navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000) in your web browser.

### 🌟 Dashboard Features:
* **Webcam Mode:** Scan your face live using your webcam.
* **Upload Area:** Drag & drop single or batches of images.
* **Demographic Reports:** Interactive pie charts and histograms showing age distribution and gender split generated dynamically using Chart.js.
* **Export Reports:** Download a comprehensive session report detailing all processed faces as a text file.
