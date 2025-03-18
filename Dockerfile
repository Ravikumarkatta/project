# project/Dockerfile
# Use an official Python runtime as the base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . .

# Expose ports for monitoring services
EXPOSE 8000  
EXPOSE 8050  

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the application (will be overridden in docker-compose.yml)
CMD ["python", "src/data/pipeline.py"]