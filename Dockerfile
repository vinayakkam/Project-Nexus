FROM python:3.11-slim

# System libraries:
#  - libgl1/libglib2.0-0/etc: OpenCV, needed by rapidocr-onnxruntime
#  - libgomp1: OpenMP runtime, needed by llama-cpp-python's compiled libllama.so
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgl1-mesa-glx \
    libgthread-2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    libx11-xcb1 \
    libxcb-render0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080
EXPOSE 8080

# Single worker: the model lives in one process's memory, multiple gunicorn
# workers would each load their own separate copy.
CMD exec gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 600 app:app
