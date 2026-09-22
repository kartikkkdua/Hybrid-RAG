# Backend image. Builds the React frontend, then serves API + static files.
FROM node:20-slim AS frontend
WORKDIR /fe
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ ./app/
COPY eval/ ./eval/
COPY mcp_server/ ./mcp_server/
COPY data/ ./data/
# built frontend served by FastAPI at /
COPY --from=frontend /fe/dist ./frontend/dist

EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
