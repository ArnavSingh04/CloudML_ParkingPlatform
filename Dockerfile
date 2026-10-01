# SmartPark API service image.
# NOTE: the model weights and supplied images are NOT copied in — they are
# mounted at runtime (see docker-compose.yml) via MODEL_PATH / volumes.
FROM python:3.12-slim

# System libraries required by ultralytics/opencv at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/models/model.pt

COPY requirements.txt .
# Install the CPU-only torch/torchvision FIRST so ultralytics doesn't drag in
# the default Linux CUDA wheels (~6-8GB of nvidia-* runtime). Inference is CPU.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1 torchvision==0.20.1 \
    && pip install --no-cache-dir -r requirements.txt

# Only the application code — never the model or images.
COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
