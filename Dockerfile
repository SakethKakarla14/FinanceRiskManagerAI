# Use an official lightweight Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install critical C++ compilers and system dependencies for LightGBM/PyTorch
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Space REQUIREMENT: Create a non-root user (id 1000)
# Security protocol mandates that Docker containers must not run as Root.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Switch context to the new user's space
WORKDIR $HOME/app

# Copy requirement files first (leverages Docker layer caching)
COPY --chown=user requirements.txt .

# Install Python dependencies locally inside the User container
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire backend API, HTML frontend, and ML models
COPY --chown=user . .

# Hugging Face Spaces strictly require the application to listen on PORT 7860
EXPOSE 7860

# Force Uvicorn to run on port 7860 using 0.0.0.0 (open ingress)
# This overrides the local port 5000 setting inside server.py
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
