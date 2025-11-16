# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for Docker caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else
COPY . .

# Create db directory if it doesn't exist
RUN mkdir -p db

# Expose port 8080 (Cloud Run requirement)
EXPOSE 8080

# Set environment variables
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1

# Run the Flask app on port 8080
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=8080"]