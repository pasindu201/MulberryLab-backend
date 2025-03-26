# Use a smaller base image
FROM python:3.12.4-slim

# Set the working directory inside the container
WORKDIR /app

# Install OpenCV dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# Expose port 8080 (same as in docker-compose)
EXPOSE 8080

# Run Flask app
CMD ["python", "app.py"]    
# multistage build
