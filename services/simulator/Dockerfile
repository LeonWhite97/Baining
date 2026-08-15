FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /simulator
COPY services/simulator/pyproject.toml ./pyproject.toml
COPY services/simulator/simulator ./simulator
RUN pip install --no-cache-dir .

CMD ["python", "-m", "simulator.main"]
