# Dockerfile

# Use Python 3.12 slim image for a smaller footprint
FROM python:3.12-slim

# Install system dependencies (adjust as needed for your project)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose the port the app runs on (adjust if needed)
EXPOSE 8000

# Command to run the application (adjust if your entry point is different)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
