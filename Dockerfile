# Dockerfile for APL 2025 MLP project with TensorFlow and all ML dependencies
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (required for some packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install \
        pandas \
        numpy \
        scikit-learn \
        tensorflow \
        xgboost \
        optuna \
        scikeras

# Default command
CMD ["python", "main_blend_mlp.py"]
