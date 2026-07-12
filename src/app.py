import os
import sys
import io
import base64
import cv2
print("DEBUG - cv2 attributes:", dir(cv2))
print("DEBUG - cv2 file path:", getattr(cv2, '__file__', 'no __file__'))
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import torch

# Add current folder to sys.path to resolve sister modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference import AgeGenderInference

app = FastAPI(title="Facial Age & Gender Analyzer API")

# Enable CORS for external frontend local connections if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals
model_runner = None
BACKBONE = "resnet18"
CHECKPOINT_PATH = f"models/best_age_gender_{BACKBONE}.pth"

def load_model():
    global model_runner
    if os.path.exists(CHECKPOINT_PATH):
        try:
            model_runner = AgeGenderInference(CHECKPOINT_PATH, backbone=BACKBONE)
            print("Successfully loaded trained PyTorch model.")
        except Exception as e:
            print(f"Error loading model: {e}")
            model_runner = None
    else:
        print(f"No checkpoint found at {CHECKPOINT_PATH}. Inference will fall back to mock predictions.")
        model_runner = None

@app.on_event("startup")
def startup_event():
    load_model()

@app.get("/api/status")
def get_status():
    model_exists = os.path.exists(CHECKPOINT_PATH)
    metrics = None
    if model_exists:
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'))
            metrics = {
                "val_loss": round(checkpoint.get("val_loss", 0), 4),
                "val_age_mae": round(checkpoint.get("val_age_mae", 0), 2),
                "val_gender_acc": round(checkpoint.get("val_gender_acc", 0), 2),
                "epoch": checkpoint.get("epoch", 0)
            }
        except Exception as e:
            print(f"Error reading metrics from checkpoint: {e}")
            
    return {
        "model_trained": model_exists,
        "backbone": BACKBONE,
        "checkpoint_path": CHECKPOINT_PATH,
        "metrics": metrics
    }

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file format")
        
    global model_runner
    # Re-verify and load trained model if checkpoint is generated post-startup
    if model_runner is None and os.path.exists(CHECKPOINT_PATH):
        load_model()
        
    if model_runner is not None:
        predictions = model_runner.predict_faces(img)
        is_mock_flag = False
    else:
        # Mock mode if model is not trained yet
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
        
        predictions = []
        np.random.seed(len(contents) % 1000)
        for i, (x, y, w, h) in enumerate(faces):
            mock_gender = "Female" if np.random.rand() > 0.5 else "Male"
            mock_age = float(np.random.randint(18, 55) if i % 2 == 0 else np.random.randint(2, 80))
            predictions.append({
                "box": [int(x), int(y), int(w), int(h)],
                "age": mock_age,
                "gender": mock_gender,
                "gender_confidence": round(80.0 + np.random.rand() * 19.0, 1)
            })
        is_mock_flag = True
            
    # Draw boxes & labels on image
    annotated_img = img.copy()
    for pred in predictions:
        x, y, w, h = pred["box"]
        age = pred["age"]
        gender = pred["gender"]
        
        color = (235, 75, 75) if gender == "Male" else (235, 75, 180)  # custom BGR colors
        
        cv2.rectangle(annotated_img, (x, y), (x+w, y+h), color, 3)
        label = f"{gender}, {int(age)}y"
        
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
        cv2.rectangle(annotated_img, (x, y - text_height - 10), (x + text_width + 10, y), color, -1)
        cv2.putText(annotated_img, label, (x + 5, y - 5), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
    _, buffer = cv2.imencode('.jpg', annotated_img)
    encoded_img = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "faces_detected": len(predictions),
        "predictions": predictions,
        "annotated_image": f"data:image/jpeg;base64,{encoded_img}",
        "is_mock": is_mock_flag
    }

# Serves static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def get_home():
    return FileResponse("static/index.html")
