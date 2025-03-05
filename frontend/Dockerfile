# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install Poetry
RUN pip install poetry

# Disable Poetry virtual environments (forces system-wide installation)
RUN poetry config virtualenvs.create false

# Copy dependency
COPY ./pyproject.toml ./

# Install dependencies
RUN poetry install --no-root

# Copy application code
COPY ./app ./app

# Expose Streamlit's default port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]

