FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install OS deps (optional, for numpy/pandas)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask port
EXPOSE 5000

# Run app
CMD ["python", "-m", "flask", "--app", "app.app", "run", "--host=0.0.0.0", "--port=5000"]