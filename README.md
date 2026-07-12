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

---

## 🌐 Deployment

This application is ready for cloud deployment. We have prepared an optimized [Dockerfile](file:///Users/abishekks/.gemini/antigravity/scratch/facial_demographics/Dockerfile) that installs system dependencies for OpenCV and optimizes PyTorch to run on lightweight CPU-only resources.

### Option 1: Hugging Face Spaces (Recommended for Machine Learning apps)
1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces).
2. Choose **Docker** as the SDK (select the **Blank** template).
3. Clone the Space repository and copy the project files there (or connect it to your GitHub).
4. Upload the model file `models/best_age_gender_resnet18.pth` directly through the Hugging Face web interface (as it exceeds GitHub's 100MB direct file limit).

### Option 2: Render / Railway / Google Cloud Run
1. Connect your GitHub repository to [Render](https://render.com) or [Railway](https://railway.app).
2. Create a new **Web Service** and choose **Docker** as the environment.
3. Add the `PORT` environment variable (the Dockerfile will automatically bind to it).

### Deployed Application Link
👉 **Live Demo:** [https://abishek-gitit-demogface-ai.hf.space](https://huggingface.co/spaces/Abishek-gitit/DemogFace-Ai) *(Replace with your live URL once active)*

