# SwitchLLM Console, ready to host: docker build -t switchllm . && docker run -p 8000:8000 switchllm
# For real model answers, pass -e OPENROUTER_API_KEY=... and set SWITCHLLM_LIVE=1.
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
ENV HOST=0.0.0.0 PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "exec switchllm demo --host \"$HOST\" --port \"$PORT\" ${SWITCHLLM_LIVE:+--live}"]
