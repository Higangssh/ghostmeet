FROM python:3.12-slim

# install ffmpeg for whisper's audio processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install python dependencies first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# pre-download whisper model at build time (so first run is instant)
ARG WHISPER_MODEL=base
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('${WHISPER_MODEL}', device='cpu')"

# copy application code
COPY backend/ backend/

EXPOSE 8877

ENV GHOSTMEET_MODEL=${WHISPER_MODEL}
ENV GHOSTMEET_DEVICE=cpu
ENV GHOSTMEET_CHUNK_INTERVAL=300

CMD ["python", "-m", "backend"]
