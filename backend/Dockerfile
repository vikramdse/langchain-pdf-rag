# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y curl

# Install Poetry
RUN pip install poetry

# Disable Poetry virtual environments (forces system-wide installation)
RUN poetry config virtualenvs.create false

# Copy dependency files
COPY ./pyproject.toml ./

# Install dependencies
RUN poetry install --no-root

# Copy application code
COPY ./app ./app
COPY ./uploads ./uploads
COPY ./data ./data

# Create necessary directories
RUN mkdir -p /app/uploads /app/data/chroma_db

# Expose FastAPI's default port
EXPOSE 8000

# Run FastAPI with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
