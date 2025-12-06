FROM python:3.12-slim

WORKDIR /app

# Install system dependencies and apply security updates
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy and set up entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Use exec form for proper signal handling
CMD ["/entrypoint.sh"]
