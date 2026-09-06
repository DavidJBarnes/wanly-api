FROM python:3.12-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ---

FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY app/ app/
COPY alembic/ alembic/
COPY alembic.ini .

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app"

USER appuser

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# --timeout-keep-alive MUST exceed the daemon's httpx pool expiry (30s, wanly-gpu-daemon
# queue_client.py). uvicorn's default is 5s, so every worker was handed pooled sockets the
# server had already closed and found out only after writing the request -- a steady drip of
# RemoteProtocolError on /segments/next and /workers/*/heartbeat (#262).
#
# Server longer than client is the correct direction: whichever end expires first decides,
# and that should be the end which can retire a connection with no request in flight.
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8001 --timeout-keep-alive 65"]
