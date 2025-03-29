# First Stage: Build dependencies
FROM python:3.12.4-slim AS builder

WORKDIR /app

# Install OpenCV and system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies in a temporary layer
COPY requirements.txt /app/
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Second Stage: Final image (lighter)
FROM python:3.12.4-slim

WORKDIR /app

# Install OpenCV runtime dependencies in the final image
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy only the installed dependencies from the builder stage
COPY --from=builder /install /usr/local

# Copy the rest of the application code
COPY . /app/

# Expose port 8080
EXPOSE 8080

# Run Flask app
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
