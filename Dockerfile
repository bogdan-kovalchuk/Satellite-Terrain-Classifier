FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps (curl for uv installer, libgl/libjpeg not required for PIL in most cases,
# but pillow may need some libs depending on wheels; keep minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /code

# Copy dependency files first (better Docker layer caching)
COPY pyproject.toml uv.lock* ./

# Install dependencies into a project venv
RUN /root/.local/bin/uv sync --frozen --no-dev

# Copy application code
COPY app ./app
COPY src ./src

# Copy model artifact (required because app loads model at import/startup)
COPY model ./model

EXPOSE 8000

CMD ["/root/.local/bin/uv", "run", "--no-sync", "uvicorn", "app.main:app", "--host=0.0.0.0", "--port=8000"]
