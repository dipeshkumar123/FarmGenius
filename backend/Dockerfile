# Multi-stage, slim Python image for optimal performance
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies first (caching layer)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy remaining source code
COPY . .

# Expose port and run the application
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
