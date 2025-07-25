FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements_gpu.txt .
RUN pip3 install --no-cache-dir -r requirements_gpu.txt

# Copy application files
COPY . .

# Make startup script executable
RUN chmod +x start.sh

# Run the application
CMD ["./start.sh"]
