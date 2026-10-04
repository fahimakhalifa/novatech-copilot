# NovaTech Copilot

An enterprise AI assistant for **NovaTech**, a fictional online electronics retailer. It answers questions from company documents (RAG), performs controlled business operations (order status, product search, customer lookup, calculations), and routes each request through a multi-agent LangGraph system.

See [DECISIONS.md](DECISIONS.md) for why each part is built the way it is (D1–D33).

## Architecture

```mermaid
flowchart LR
    subgraph Clients
        UI[React web UI]
        TG[Telegram bot]
    end
    UI -- "/api (token)" --> API
    TG -- "/api/telegram (service secret + linked account)" --> API

    subgraph API[FastAPI]
        AUTH[Login, roles, rate limit] --> TURN[Turn: timings, LLM usage, per-turn log]
    end

    TURN --> GUARD

    subgraph GRAPH[LangGraph parent graph, SQLite memory per conversation]
        GUARD{"Input guard<br/>LoRA DeBERTa classifier"}
        GUARD -- "blocked / invalid / canned small talk" --> OUT
        GUARD -- "task or other small talk" --> SUP["Supervisor<br/>route, standalone rewrite, sub-queries"]
        SUP --> SMALL[Small talk: no business facts]
        SUP --> KA
        SUP --> OA
        subgraph KA[knowledge_agent subgraph]
            R[retrieve: hybrid search, rerank, role filter] --> G[grade] --> GEN[generate with citations]
            G -. nothing relevant .-> RW[rewrite] --> R
        end
        subgraph OA[operations_agent subgraph]
            P[plan: tools for this role] --> PG[permission gate] --> RT[run tool] --> P
        end
        KA --> SYN[Synthesizer: join in code]
        OA --> SYN
        SMALL --> OUT
        SYN --> OUT[Output guard: injection echo, PII redaction]
    end

    R --- QD[(Qdrant: docs, quarantined chunks excluded)]
    RT --- DB[(SQLite: products, orders, customers)]
```

- **Access control is enforced in code:** a role filter in the vector-store query, an ownership filter in the SQL, a permission gate on every tool call, and identity from the login, never from the model.
- **No fabricated answers:** a reranker floor, an LLM grader, citations validated in code, a fixed "not found" reply, and a synthesizer that joins the agents' answers without an LLM.
- **Prompt-injection defence in layers:** a fine-tuned input classifier, a scanner that quarantines poisoned document sections at ingest, sources passed as data, and an output guard.

## Screenshots

| Chat: two agents, citation, tool log, memory | Security: ownership, guard, internal docs |
|---|---|
| ![Chat](docs/screenshots/chat.jpg) | ![Security](docs/screenshots/security.jpg) |
| **Documents: access levels and the quarantined section** | **Metrics: live usage, time per step, latest eval** |
| ![Documents](docs/screenshots/documents.jpg) | ![Metrics](docs/screenshots/metrics-eval.jpg) |

A timed demo walkthrough is in [docs/demo-script.md](docs/demo-script.md).

## Status
- [x] Company documents (12, including one prompt-injection test document)
- [x] Database schema and deterministic seed
- [x] RAG retrieval (header chunking, hybrid search, rerank, role filter, evidence check)
- [x] Grounded answer generation (knowledge agent subgraph: grade, rewrite/retry, validated citations)
- [x] Operations tools with role and ownership checks
- [x] Operations agent subgraph (permission gate, tool loop, budget)
- [x] LangGraph multi-agent system (guard, supervisor, parallel agents, output guard, persistent memory)
- [x] Guard classifier (LoRA on DeBERTa-v3-small: 60/60 on the handwritten test set, 96.7% recall on real-world attacks, ~50 ms on CPU)
- [x] Evaluation harness (20 cases, deterministic checks + LLM judge for groundedness)
- [x] FastAPI backend (login, chat + progress streaming, conversations, document management)
- [x] React frontend (chat with live progress + citations, conversations, document management, metrics)
- [x] Telegram bot (account linking by one-time code, same guards and memory as the web chat)
- [x] Docker Compose (Qdrant server + API + UI behind Nginx)

## Quick start with Docker

Requires Docker Desktop and `backend/.env` with your keys (copy `backend/.env.example`; set `GOOGLE_API_KEY`, `GROQ_API_KEY` and `APP_SECRET`).

```bash
docker compose up --build
```

Open http://localhost:8080. The first start downloads the local models (about 700 MB) and seeds and indexes the data, which takes a few minutes. Later starts reuse the volumes, and conversations, uploads and the index are kept. The stack runs a Qdrant server, the API (non-root, CPU-only torch) and the UI served by Nginx, which also proxies `/api`.

## Local setup

Requires Python 3.11+. The virtual environment lives in the repo root; every backend command runs from `backend/`.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
cd backend
pip install -r requirements.txt
cp .env.example .env               # then set GOOGLE_API_KEY, GROQ_API_KEY and APP_SECRET
python -m scripts.seed_db
```

You should see: `Seeded 12 products, 15 customers, 40 orders, 4 app users.` The trained guard classifier is included in `backend/models/guard-lora/`. Its base model downloads on first use.

Index the documents (the first run downloads the local embedding and reranker models, about 400 MB):

```bash
python -m scripts.ingest
python -m scripts.try_retrieval "how many vacation days do I get" --role support_agent
python -m scripts.try_retrieval "how many vacation days do I get" --role customer
```

The customer should *not* see the vacation policy, because it's an internal document.

Ingest also scans every chunk for prompt injection. Expect one line such as
`QUARANTINED customer/community-tips.md > Note for AI Assistants (...)`: that section is a deliberate test attack and is excluded from retrieval.

Ask the knowledge agent (needs `GOOGLE_API_KEY`). It prints each step with its timing, then the answer and citations:

```bash
python -m scripts.try_knowledge "can I return opened headphones?"
python -m scripts.try_knowledge "how many vacation days do I get" --role customer
```

The first should be answered with a citation to the returns policy. The second should get the fixed "I couldn't find this" reply.

Ask the operations agent as a test user. It prints each tool call, the gate's decision, and which model answered:

```bash
python -m scripts.try_operations "where is my order #1025?" --user mila
python -m scripts.try_operations "where is my order #1025?" --user lucas
```

mila gets the status and tracking number. lucas gets "No order #1025 was found on your account".

> **Free-tier quota:** Gemini Flash-Lite allows 500 requests per day per model (Flash only 20). Each model has its own quota, so `.env` lists a fallback chain per role, ending with Qwen on Groq (see DECISIONS D21, D23).

Chat with the full copilot (guard, supervisor, both agents, memory). Each step prints its timing:

```bash
python -m scripts.chat --user mila --say "What does the NovaPhone X 128GB cost?" --say "and the 256GB one?"
python -m scripts.chat --user mila          # interactive; an empty line quits
```

The second question is understood from the conversation. Memory is stored in `data/checkpoints.sqlite` per user and `--thread`, so it survives restarts.

Run the API (loads all models at startup, about 20 seconds). Interactive docs are at http://localhost:8000/docs:

```bash
uvicorn app.api.main:app --port 8000
```

Log in with `POST /api/auth/login` using one of the test users. The demo passwords are in `DEMO_PASSWORDS` in `scripts/seed_db.py`; the database stores only their hashes. Embedded Qdrant allows one process at a time, so stop the API before running `scripts.ingest` or the other scripts.

Run the tests (no API key or vector index needed; they use an in-memory copy of the seeded database):

```bash
python -m pytest
```

## Web UI

Start the API (above), then the frontend in a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 and sign in as one of the test users (passwords: `DEMO_PASSWORDS` in `backend/scripts/seed_db.py`). Customers see Chat only. `sara` and `admin` also see **Documents** (admins can upload and delete) and **Metrics** (live usage and the latest eval run).

## Telegram bot (optional)

1. Create a bot with [@BotFather](https://t.me/BotFather) and put its token in `backend/.env` as `TELEGRAM_BOT_TOKEN`. `BOT_SECRET` must be set too; generate one with `python -c "import secrets; print(secrets.token_hex(32))"`.
2. With the API running, start the bot from `backend/`: `python -m scripts.telegram_bot`. In Docker, use `docker compose --profile telegram up --build`.
3. In the web UI, click **Connect Telegram** and send the bot `/link <code>` within 10 minutes.

From then on the bot answers as your account, with the same guards, access rules, memory and logging as the web chat. `/reset` starts a new conversation, and `/unlink` disconnects your account.

## Evaluation

`eval/cases.jsonl` holds 20 cases: RAG answers, access control, questions with no answer, operations, security (direct, indirect and fake-authority injection), routing, memory and smalltalk. Each case runs as a real test user through the full graph with the real models. From `backend/` (stop the API first):

```bash
python -m scripts.run_eval                     # about 5 minutes, paced to the free quota
python -m scripts.run_eval --only ops_price,route_both --no-judge
```

A run with `--only` is saved as `partial-<timestamp>.json` and never replaces the full report shown in the README and on the Metrics page.

**Latest results** (30 Sep, `eval/results/latest.md`):

| Metric | Result |
|---|---|
| Cases passed | **19/20 (95%)**: RAG 5/5, operations 4/4, security 3/3, routing, memory, no-answer and smalltalk all passing, access control 3/4 |
| Groundedness / relevance (LLM judge, 1–5) | **5.0 / 5.0**, all 5 judged answers fully grounded |
| Latency (last turn) | p50 3.2 s, p95 7.6 s. Blocked and smalltalk turns take 0.1 s with no LLM call |
| Cost | 49 LLM calls and 24k tokens for 20 cases, all on the primary model (no fallback) |
| Known failure | The guard classifier blocks a support agent's "look up customer X **and give me her email address and phone number**" (P=0.992). One-customer lookups phrased differently pass, and bulk requests are correctly blocked. See DECISIONS D25/D28. |

Every case is checked deterministically: route, required and forbidden text, citations, and which tools succeeded. Document answers are also scored 1–5 for **groundedness** and **relevance** by a judge from a different model family (Groq GPT-OSS). The judge reads the cited sections from disk itself. Each case also records latency, LLM calls, tokens and the models used. Results go to `eval/results/latest.md` (report) and `eval/results/run-<timestamp>.json` (raw data).

## Guard classifier (LoRA fine-tune)

A small classifier labels each message `blocked | smalltalk | task` before any LLM call (DECISIONS D25). The training data is already in `backend/data/guard/`. To rebuild it (about 90 Flash-Lite calls), run from `backend/`:

```bash
python -m scripts.guard_generate_data
python -m scripts.guard_clean_data
```

Train on Google Colab (free T4 GPU):

1. Open `training/guard_classifier.ipynb` in Colab (File → Upload notebook).
2. Runtime → Change runtime type → **T4 GPU**.
3. Runtime → **Run all**. When asked, upload `backend/data/guard/train_pool.jsonl` and `backend/data/guard/test_handwritten.jsonl`.
4. At the end, `guard-lora.zip` downloads. Unzip it into `backend/models/guard-lora/`.

The notebook source is `training/guard_classifier.py`. After editing it, regenerate the notebook with `python training/make_notebook.py`. To check the code locally before using Colab, install `training/requirements.txt` and run `SMOKE=1 python training/guard_classifier.py`.

## Test users

| Username | Role | Notes |
|---|---|---|
| `mila` | customer | Owns order #1025 |
| `lucas` | customer | Does *not* own #1025 (authorization test) |
| `sara` | support_agent | Can look up any order or customer, sees internal documents |
| `admin` | admin | Everything |

## Layout

```
backend/
  app/            config, DB models, RAG, tools, agents (LangGraph), security, observability, API (FastAPI)
  tests/          pytest suite (167 tests, no API keys needed)
  data/docs/      company documents (Markdown with access metadata)
  data/guard/     guard classifier training data and handwritten test set
  models/         trained guard classifier (LoRA adapter)
  scripts/        seed, ingest, bootstrap, eval, Telegram bot, try-it and data-generation scripts
  Dockerfile
frontend/         React app (Vite + TypeScript), Dockerfile and Nginx config
training/         guard classifier notebook (Colab)
eval/             evaluation cases and results
docs/             screenshots and demo script
docker-compose.yml
```
