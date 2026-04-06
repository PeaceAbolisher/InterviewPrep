# Base image Python
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy and install dependencies first
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy only what the API needs (training files stay out)
COPY model.pkl .
COPY main.py .

# Document the port the app listens on
EXPOSE 8000

# Start the API server - 0.0.0.0 makes it reachable from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]