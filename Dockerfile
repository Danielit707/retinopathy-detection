# Use a lightweight, official Python runtime base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install minimal system utilities required for building basic C dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to take full advantage of Docker's layer caching
COPY requirements.txt .

# Install dependencies using the optimized CPU wheel index
RUN pip install --no-cache-dir -r requirements.txt

# Copy your source application and the trained weights into the container
COPY src/ ./src/
COPY weights/ ./weights/

# Expose the internal port that Uvicorn binds to
EXPOSE 8000

# Run the API server. We bind to 0.0.0.0 so it can accept external traffic
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]