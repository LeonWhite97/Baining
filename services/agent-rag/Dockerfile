FROM python:3.12-slim
WORKDIR /app
COPY services/agent-rag/pyproject.toml ./
COPY services/agent-rag/agent_rag ./agent_rag
RUN pip install --no-cache-dir fastapi uvicorn pydantic sqlalchemy "psycopg[binary]"
EXPOSE 8013
CMD ["uvicorn", "agent_rag.main:app", "--host", "0.0.0.0", "--port", "8013"]
