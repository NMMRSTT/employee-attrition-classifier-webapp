# Use the official Python image.
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Create and activate a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies from requirements.txt and Google Cloud Storage client
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install google-cloud-storage

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run streamlit when the container launches on port 8080
CMD ["streamlit", "run", "app.py", "--server.port", "8080", "--server.address", "0.0.0.0"]
