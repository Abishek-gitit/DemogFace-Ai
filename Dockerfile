FROM python:3.10-slim

# Install system dependencies for OpenCV and download tools
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Optimize PyTorch installation to use CPU-only wheels (saves ~1.5GB of RAM/disk space on hosting services)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Download the model weights directly from the Hugging Face Model Hub during container build
RUN mkdir -p models && \
    wget -O models/best_age_gender_resnet18.pth https://huggingface.co/Gomma456/DemogFace-model/resolve/main/best_age_gender_resnet18.pth && \
    apt-get purge -y wget && apt-get autoremove -y

EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
