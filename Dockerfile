FROM python:3.10-slim

# Install the curl package
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the model file from GitHub Releases during the BUILD phase
RUN mkdir -p models && \
    curl -L -o models/fcos_pcb_model.onnx https://github.com/tolgaguclu17/PCB-Defect-Detection-Fcos-Api/releases/download/v1.0.0/fcos_pcb_model.onnx

COPY . .

# Dynamically set Render's PORT variable (default 10000 or the specified PORT)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
