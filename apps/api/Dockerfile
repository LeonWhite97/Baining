FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY apps/api/pyproject.toml ./pyproject.toml
COPY apps/api/app ./app
COPY apps/api/alembic.ini ./alembic.ini
COPY apps/api/migrations ./migrations
RUN pip install --no-cache-dir .

EXPOSE 8000
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
