.DEFAULT_GOAL := help
PY := .venv/bin/python
PIP := .venv/bin/pip

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

venv: ## Create the virtualenv and install core deps
	python3 -m venv .venv
	$(PIP) install -q -U pip
	$(PIP) install -q -r requirements.txt

ml: ## Install optional ML backends (real embeddings + cross-encoder)
	$(PIP) install -r requirements-ml.txt

agent: ## Install the LangGraph multi-agent layer
	$(PIP) install -r requirements-agent.txt

agent-demo: ## Run a multi-part question through the agent graph
	$(PY) -m app.cli agent "How does a cross-encoder differ from a bi-encoder, and what does nDCG capture?"

pg: ## Install the Postgres driver and start a pgvector database
	$(PIP) install -r requirements-pg.txt
	docker compose --profile pg up -d db
	@echo "pgvector on :55432 — run with DB_BACKEND=postgres"

pg-test: ## Run the Postgres backend tests against the local pgvector db
	PG_TEST_DSN=postgresql://postgres:postgres@localhost:55432/rag $(PY) -m pytest tests/test_pg_store.py -q

serve-pg: ## Run the API against Postgres + pgvector
	DB_BACKEND=postgres PG_DSN=postgresql://postgres:postgres@localhost:55432/rag \
		$(PY) -m uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

ingest: ## Ingest the sample corpus
	$(PY) -m app.cli ingest data/sample_docs

demo: ## Ingest + a sample search and answer
	$(PY) -m app.cli ingest data/sample_docs
	$(PY) -m app.cli search "how does RRF combine scores?"
	$(PY) -m app.cli ask "what constant does RRF use?"

eval: ## Run the retrieval evaluation + config table (JUDGE=1 adds RAGAS metrics)
	$(PY) -m eval.run_eval --docs data/sample_docs $(if $(JUDGE),--judge,)

ab: ## Run the prompt/model A/B harness
	$(PY) -m eval.ab_harness --docs data/sample_docs

mcp-demo: ## Run the MCP client<->server round-trip demo
	$(PY) -m mcp_server.client

test: ## Run the test suite
	$(PY) -m pytest -q

serve: ## Run the API (http://localhost:8000)
	$(PY) -m uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

frontend: ## Run the Vite dev server (http://localhost:5173)
	cd frontend && npm install && npm run dev

build-frontend: ## Build the frontend into frontend/dist
	cd frontend && npm install && npm run build

docker: ## Build and run everything in Docker
	docker compose up --build

.PHONY: help venv ml agent agent-demo pg pg-test serve-pg ingest demo eval ab mcp-demo test serve frontend build-frontend docker
