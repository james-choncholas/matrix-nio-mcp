FROM python:3.12-slim

# libolm3: runtime E2EE library; libolm-dev + gcc: needed to build python-olm wheel
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libolm3 libolm-dev gcc curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
# Install deps first (layer cached until pyproject.toml changes)
RUN pip install --no-cache-dir -e ".[e2e]"

COPY src/ src/

# Nio E2EE store is mounted here at runtime
VOLUME ["/data/nio_store"]

EXPOSE 8000

CMD ["python", "-m", "nio_mcp.server"]
