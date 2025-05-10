# Use an official Python runtime as a parent image
# Using a slim version for a smaller image size. Bullseye is a stable Debian release.
FROM python:3.10-slim-bullseye AS builder

# Set environment variables
# PYTHONUNBUFFERED: Ensures print statements and logs are sent straight to terminal without being buffered first.
# PIP_NO_CACHE_DIR: Disables pip cache, reducing image size.
# PIP_DISABLE_PIP_VERSION_CHECK: Disables pip version check, speeding up pip commands.
# PIP_DEFAULT_TIMEOUT: Increases pip timeout.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# - build-essential: For compiling Python packages that have C extensions if no pre-compiled wheel is available.
# - ffmpeg: Required by your application for video processing.
# - git: In case any pip packages are installed directly from git (though not in your current requirements.txt, it's good practice).
# --no-install-recommends: Reduces installed packages to only essential ones.
# Clean up apt cache to reduce image size.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    git \
    curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
# This is done before copying the rest of the code to leverage Docker's layer caching.
# If requirements.txt doesn't change, this layer won't be rebuilt.
COPY requirements.txt .

# Install Python dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# --- Final Stage ---
# Use a slim base image again for the final image
FROM python:3.10-slim-bullseye AS final

# Set environment variables for the final image
ENV PYTHONUNBUFFERED=1 \
    APP_HOME=/app \
    # Default paths inside the container. These should match what your app expects
    # or be configurable via runtime environment variables.
    TEMP_VIDEO_DIR="/app/temp_uploads" \
    CHROMA_DB_PATH="/app/data/chroma_db_dev"

# Set the working directory
WORKDIR $APP_HOME

# Create necessary directories for the application
# These paths will be used by your application for storing temporary files and ChromaDB data.
# In docker-compose, these paths will be mapped to Docker volumes for persistence.
RUN mkdir -p $APP_HOME/temp_uploads \
    && mkdir -p $APP_HOME/data/chroma_db_dev \
    && mkdir -p $APP_HOME/services # Ensure services directory structure is created

# Install FFmpeg (needed at runtime)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from the builder stage to the final stage.
# This copies the site-packages directory where pip installed everything.
COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/
# Copy any executables installed by pip (like uvicorn)
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy the application source code into the container
COPY main.py .
COPY services/ $APP_HOME/services/
# Note: .env file is NOT copied by default. It should be provided at runtime.

# Create a non-root user and group for security
RUN groupadd -r appgroup && useradd --no-log-init -r -g appgroup appuser

# Change ownership of the app directories to the non-root user
# This allows the application to write to these directories if needed (e.g., ChromaDB, temp files).
RUN chown -R appuser:appgroup $APP_HOME

# Switch to the non-root user
USER appuser

# Expose the port that Uvicorn will run on (must match the port in CMD)
EXPOSE 8001

# Define the command to run your application using Uvicorn
# --host 0.0.0.0 makes the server accessible from outside the container.
# --port 8001 matches the EXPOSE instruction.
# reload is typically disabled for production images.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]