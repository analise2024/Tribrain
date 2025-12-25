# --- Stage 1: Builder ---
FROM python:3.11-slim-bookworm AS builder

# Evitar a criação de arquivos .pyc e habilitar o log em tempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar dependências de compilação
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Criar ambiente virtual para isolar dependências
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Instalar dependências primeiro para aproveitar o cache do Docker
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .[dev,lora,vlm]

# --- Stage 2: Runtime ---
FROM python:3.11-slim-bookworm AS runtime

LABEL maintainer="Tribrain Maintainers"
LABEL description="Cloud Native Control Layer for World Models"

WORKDIR /app

# Copiar apenas o ambiente virtual e o código necessário do builder
COPY --from=builder /opt/venv /opt/venv
COPY src/ ./src/
COPY pyproject.toml .
COPY README.md .

# Instalar dependências de runtime necessárias (ex: OpenCV precisa de libGL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Configurar variáveis de ambiente
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app/src"

# Criar usuário não-root para segurança
RUN useradd -m tribrain && chown -R tribrain:tribrain /app
USER tribrain

# Volume para persistência local (opcional se usar S3)
VOLUME ["/app/outputs"]

# Comando padrão
ENTRYPOINT ["python", "-m", "tribrain.cli"]
CMD ["--help"]
