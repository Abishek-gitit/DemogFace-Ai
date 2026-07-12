import os
import cv2
from PIL import Image
import torch
from torchvision import transforms
from model import AgeGenderModel

class AgeGenderInference:
    def __init__(self, checkpoint_path, backbone="resnet18", device=None):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
            
        print(f"Initializing inference on device: {self.device}")
        
        # Load architecture
        self.model = AgeGenderModel(backbone=backbone, pretrained=False)
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.img_size = checkpoint.get('img_size', 128)
        
        # Preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load Haar Cascade face detector
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load OpenCV face detector Haar cascade.")
        
    def predict_faces(self, cv2_img):
        # Convert BGR (OpenCV format) to Grayscale for detection
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(40, 40)
        )
        
        results = []
        
        for (x, y, w, h) in faces:
            # Crop face area
            face_crop = cv2_img[y:y+h, x:x+w]
            if face_crop.size == 0:
                continue
                
            # Convert cropped face to RGB for PIL & PyTorch
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_rgb)
            
            # Apply transforms
            tensor_img = self.transform(pil_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                age_pred, gender_pred = self.model(tensor_img)
                
                # Age value
                age = float(age_pred.item())
                age = max(0.0, age)
                
                # Gender value & confidence
                gender_probs = torch.softmax(gender_pred, dim=1)
                gender_idx = int(torch.argmax(gender_probs).item())
                gender_conf = float(gender_probs[0, gender_idx].item())
                gender = "Male" if gender_idx == 0 else "Female"
                
            results.append({
                "box": [int(x), int(y), int(w), int(h)],
                "age": round(age, 1),
                "gender": gender,
                "gender_confidence": round(gender_conf * 100, 1)
            })
            
        return results
