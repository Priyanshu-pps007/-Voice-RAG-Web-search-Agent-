# Voice With RAG

Voice-first multi-agent RAG assistant with:

- FastAPI backend
- LangChain/LangGraph-style agent orchestration
- Qdrant vector database with HNSW indexing
- Ollama local embeddings via `bge-m3`
- Tavily-powered web search agent
- Next.js frontend with browser speech recognition, streamed responses, text-to-speech playback, and barge-in interruption

This project is designed as a conversational assistant that can decide between:

1. Answering directly from the LLM
2. Retrieving from private uploaded PDFs
3. Searching the public web for fresh information

## What This Project Does

The system combines private document retrieval and live web search inside a single voice-oriented interface.

Users can upload PDFs, which are parsed, chunked, embedded, and stored in Qdrant. During conversation, the main agent decides whether the question should be answered:

- directly by the model,
- through the private RAG pipeline, or
- through a dedicated web-search agent.

On the frontend, the browser continuously listens, streams speech into text, sends the final utterance over WebSocket, receives token-level responses from the backend, and speaks the answer back sentence by sentence. If the user starts speaking while the assistant is talking, the frontend can interrupt the current response and resume listening.

## Core Stack

### Backend

- FastAPI
- LangChain `create_agent`
- LangGraph in-memory checkpointing
- Groq-hosted `llama-3.3-70b-versatile`
- Ollama embeddings with `bge-m3`
- Qdrant vector database
- Tavily web search

### Frontend

- Next.js 16
- React 19
- Tailwind CSS v4
- Browser WebSocket API
- Browser `SpeechRecognition` / `webkitSpeechRecognition`
- Browser `speechSynthesis`
- `getUserMedia` + `AudioContext` + `AnalyserNode` for barge-in detection

## High-Level Architecture

```text
User speech
   |
   v
Browser speech recognition
   |
   v
Next.js frontend
   |
   | WebSocket: /ws/agent
   v
FastAPI backend
   |
   v
Main routing agent
   |-------------------------------> Direct LLM answer
   |
   |----> RAG agent ----> RAG tool ----> Ollama embed(query)
   |                                  -> Qdrant filtered vector search
   |                                  -> retrieved chunks -> final answer
   |
   |----> Search agent -> Tavily tool -> public web results -> final answer
   |
   v
Token stream back to frontend
   |
   v
Sentence buffer + browser TTS
```

## Backend Architecture

The backend entrypoint is [`main.py`](./main.py). It exposes three main interaction surfaces:

- `GET /health`
- `POST /run-agent`
- `POST /rag/upload`
- `WebSocket /ws/agent`

### 1. Health Endpoint

`GET /health` returns a simple `{ "status": "ok" }` response and is intended for quick service availability checks.

### 2. Streaming Agent Endpoint

`POST /run-agent` returns a `StreamingResponse` that streams the main agent’s output as plain text. This is the simpler HTTP streaming path.

### 3. PDF Upload + Ingestion Endpoint

`POST /rag/upload` accepts up to 5 PDF files at once. For each file, the backend:

1. validates the extension,
2. writes the file into `RAG/uploads/`,
3. calls the ingestion pipeline,
4. returns file-level chunk counts and completion metadata.

The endpoint currently uses:

- `MAX_UPLOAD_FILES = 5`
- `ALLOWED_UPLOAD_SUFFIXES = {".pdf"}`

### 4. Real-Time Voice Agent WebSocket

`/ws/agent` is the primary real-time interface used by the frontend.

It supports:

- starting a new query,
- interrupting an in-progress response,
- streaming token events back to the browser,
- returning status events such as `started`, `interrupted`, `done`, and `error`.

This gives the frontend fine-grained control over low-latency voice behavior.

## Multi-Agent Design

The main orchestration logic lives in [`AgentNetwork/build_agent.py`](./AgentNetwork/build_agent.py).

This file defines three agents:

### Main Agent

The main agent is the user-facing router. Its prompt instructs it to choose among:

- direct answering,
- `RAG_agent` for private document knowledge,
- `Search_agent` for public or current web knowledge.

This is the central intelligence layer of the system. Instead of always calling tools, it is explicitly instructed to choose the cheapest sufficient path:

- direct answer first,
- RAG when private documents are needed,
- web search when current/public data is needed.

### RAG Agent

The RAG agent is a specialist agent that must use the RAG tool for every query it handles. It is constrained to answer only from the retrieved internal document context.

### Search Agent

The search agent is a web research specialist that must use Tavily search for factual public lookup tasks. It is designed for fresh or changing information such as:

- news,
- current facts,
- recent events,
- live public knowledge.

## Agent Routing Policy

The routing prompt already captures an important architectural decision: tool use is not the default.

That means the system is not a naive “always search, always retrieve” pipeline. Instead, it behaves more like a lightweight hierarchical agent system:

- `Main_agent` handles intent selection
- `RAG_agent` handles private retrieval-grounded answering
- `Search_agent` handles public web-grounded answering

This reduces unnecessary retrieval cost and keeps direct conversational turns fast.

## RAG Architecture

The RAG implementation is split mainly across:

- [`RAG/config.py`](./RAG/config.py)
- [`RAG/ingest_doc.py`](./RAG/ingest_doc.py)
- [`tools/my_tool.py`](./tools/my_tool.py)

### Vector Database: Qdrant

Qdrant is used as the vector store for document chunks.

Collection configuration:

- collection name: `RAG_Collection`
- vector size: `1024`
- distance metric: `COSINE`
- ANN index: `HNSW`
- HNSW params:
  - `m = 16`
  - `ef_construct = 100`

This is created in `ensure_collection_ready()` if the collection does not already exist.

### Why HNSW Matters Here

Qdrant’s HNSW index is the core retrieval acceleration layer.

In practice, it gives:

- efficient approximate nearest-neighbor search,
- fast top-k retrieval over embedded PDF chunks,
- good latency/quality tradeoff for conversational retrieval workloads.

For this project, HNSW is the reason the system can remain responsive while searching semantically similar chunks instead of doing exact brute-force comparisons.

### Payload Index for User Isolation

The system also creates a payload index on `user_id`:

- field: `user_id`
- schema: `KEYWORD`

This is important because retrieval queries are filtered by `user_id` before returning chunks. That allows logical separation of data inside a single Qdrant collection.

## Document Ingestion Pipeline

The ingestion flow is implemented in [`RAG/ingest_doc.py`](./RAG/ingest_doc.py).

### Step 1: PDF Parsing

PDFs are parsed using:

- `unstructured.partition.pdf.partition_pdf`

This extracts structured PDF elements before chunking.

### Step 2: Semantic Chunking

Chunks are created with:

- `unstructured.chunking.title.chunk_by_title`

Current chunking configuration:

- `max_characters = 500`
- `new_after_n_chars = 400`
- `overlap = 0`

This produces relatively compact sections suited for retrieval.

### Step 3: Embedding Generation

Each chunk is embedded through Ollama using:

- model: `bge-m3`

This happens locally through `ollama.embed(...)`.

### Step 4: Vector Storage in Qdrant

Each chunk is stored as a Qdrant point with:

- generated point id
- embedding vector
- payload:
  - `user_id`
  - `text`
  - `metadata`
  - `metadata.source_file`

### Ingestion Output

The ingestion path returns:

- `file_name`
- `chunk_count`
- `user_id`
- aggregate totals across uploaded files

## Retrieval Flow

The runtime RAG tool is implemented in [`tools/my_tool.py`](./tools/my_tool.py).

When the RAG tool is called:

1. the user query is embedded with `bge-m3`,
2. Qdrant performs vector search,
3. results are filtered by `user_id`,
4. the top 3 chunks are returned,
5. the RAG agent uses those chunks to produce the final answer.

Current search configuration:

- `limit = 3`
- vector similarity search via Qdrant
- metadata filtering through `query_filter`

The tool currently returns concatenated raw chunk text to the RAG agent, which then formats the user-facing answer.

## Web Search Agent

The public-web retrieval path is also implemented in [`tools/my_tool.py`](./tools/my_tool.py).

The `Tavily_Search` tool:

- uses `langchain_tavily.TavilySearch`
- reads `TAVILY_API_KEY` from environment
- returns up to `max_results = 2`

This search tool is only meant for public factual information and is wrapped behind the dedicated `Search_agent`.

## LangGraph / State Handling

The main agent is created with an `InMemorySaver()` checkpointer.

That means:

- the graph can retain short-lived state in memory,
- there is no persistent conversation storage layer yet,
- current session state is not durable across process restarts.

The streaming call currently uses:

- `stream_mode=["messages", "updates"]`
- `version="v2"`
- `subgraphs=True`

Only root-level model tokens are emitted back to the frontend for real-time display and TTS.

## Frontend Voice Architecture

The frontend lives in [`frontend/vrag-frontend`](./frontend/vrag-frontend).

The main voice UI is implemented in [`frontend/vrag-frontend/app/page.tsx`](./frontend/vrag-frontend/app/page.tsx).

### Core Frontend Responsibilities

The frontend handles:

- continuous microphone listening
- speech-to-text in the browser
- WebSocket communication with the backend
- upload flow for PDF ingestion
- assistant token streaming
- incremental sentence buffering
- browser text-to-speech
- barge-in interruption detection

### Browser Speech Recognition

Speech input uses:

- `window.SpeechRecognition`
- or `window.webkitSpeechRecognition`

Current behavior:

- language: `en-US`
- interim results enabled
- one final utterance is collected and then sent to the backend
- auto-resume listening is attempted after each interaction

### Streaming Response Handling

The frontend opens a WebSocket to:

- `NEXT_PUBLIC_AGENT_WS_URL`
- default: `ws://127.0.0.1:8000/ws/agent`

It consumes backend events of type:

- `status`
- `token`
- `done`
- `error`

Tokens are appended live to the assistant transcript in the UI.

### Sentence-Based TTS Buffering

The frontend does not wait for the full answer before speaking.

Instead, it:

1. accumulates streamed tokens into a buffer,
2. splits the buffer by sentence/paragraph boundaries,
3. enqueues speakable segments,
4. uses `SpeechSynthesisUtterance` to speak each segment.

This lowers perceived latency and makes the system feel more conversational.

### Barge-In / Interrupt Architecture

One of the strongest frontend features in this project is barge-in support.

The frontend uses:

- `navigator.mediaDevices.getUserMedia(...)`
- `AudioContext`
- `AnalyserNode`
- RMS energy monitoring

Current barge-in thresholds:

- RMS threshold: `0.15`
- required loud frames: `5`
- TTS grace period: `700 ms`

When the user starts speaking loudly enough while assistant TTS is active:

1. the frontend detects voice energy,
2. the current response is interrupted,
3. the frontend sends `{ action: "interrupt" }` to the backend,
4. browser speech synthesis is cancelled,
5. listening resumes for the next user turn.

This is the “frontend bridge” that makes the app feel voice-native instead of just a chat UI with audio glued on top.

## Frontend UX Surface

The UI includes:

- connection status
- listening/speaking/streaming metrics
- PDF upload stage
- transcript view for final and interim speech
- assistant streamed response panel
- controls for pause, resume, silence, reconnect, and session reset

Stylistically, the frontend uses a dark glassmorphism-like panel system with custom accent colors, status pills, radial backgrounds, and a voice-oriented dashboard layout.

## API Surface Summary

### `GET /health`

Returns backend health.

### `POST /run-agent`

Request body:

```json
{
  "query": "your question"
}
```

Returns streamed plain-text output.

### `POST /rag/upload`

Multipart form upload with:

- `files`
- optional `user_id`

Returns ingestion summary, uploaded file names, and chunk counts.

### `WS /ws/agent`

Client can send:

```json
{ "query": "Tell me about my resume" }
```

or:

```json
{ "action": "interrupt" }
```

Server sends events such as:

```json
{ "type": "status", "status": "started" }
```

```json
{ "type": "token", "token": "Hello" }
```

```json
{ "type": "done" }
```

```json
{ "type": "error", "message": "..." }
```

## Project Structure

```text
.
├── main.py                       # FastAPI app and API/WebSocket routes
├── AgentNetwork/
│   └── build_agent.py            # Main agent, RAG agent, Search agent
├── RAG/
│   ├── config.py                 # Qdrant setup and collection/index creation
│   ├── ingest_doc.py             # PDF parsing, chunking, embedding, storage
│   └── uploads/                  # Uploaded PDFs
├── tools/
│   └── my_tool.py                # RAG retrieval tool and Tavily search tool
├── db/
│   └── redis_setup.py            # Redis connection helper (not active in main flow)
└── frontend/
    └── vrag-frontend/            # Next.js voice frontend
```

## Environment Variables

Backend/runtime environment currently expects variables such as:

- `GROQ_API_KEY`
- `TAVILY_API_KEY`
- `QDRANT_CLUSTER_ENDPOINT`
- `QDRANT_API_KEY`
- `UNSTRUCTURED_API_KEY`

Frontend environment can use:

- `NEXT_PUBLIC_AGENT_WS_URL`
- `NEXT_PUBLIC_RAG_UPLOAD_URL`

Do not commit live secrets into version control. If you keep using this project, moving these values into a sanitized `.env.example` file is strongly recommended.

## Local Development

### Backend

Install Python dependencies:

```bash
uv sync
```

Run the FastAPI server:

```bash
uv run uvicorn main:app --reload
```

### Required Supporting Services

You should have access to:

- a running Qdrant instance, or a valid Qdrant Cloud endpoint
- a running Ollama server with `bge-m3` available
- valid Groq and Tavily API keys

### Frontend

```bash
cd frontend/vrag-frontend
npm install
npm run dev
```

Open:

- `http://localhost:3000`

Backend default:

- `http://127.0.0.1:8000`

## End-to-End Runtime Flow

### Document ingestion flow

1. User uploads PDF(s) from the frontend.
2. FastAPI validates and stores the files temporarily in `RAG/uploads/`.
3. `partition_pdf` extracts PDF elements.
4. `chunk_by_title` creates text chunks.
5. Ollama generates `bge-m3` embeddings.
6. Qdrant stores embeddings and payloads.
7. Files become available for retrieval.

### Voice query flow

1. User speaks in the browser.
2. Browser speech recognition captures text.
3. Final utterance is sent to `/ws/agent`.
4. Main agent decides: direct answer, RAG, or web search.
5. Specialized tool/agent path runs if needed.
6. Backend streams tokens back over WebSocket.
7. Frontend updates the transcript and speaks buffered sentences aloud.

### Barge-in flow

1. Assistant is speaking.
2. Frontend audio monitor detects user voice energy.
3. Frontend cancels TTS and sends interrupt.
4. Backend cancels the active stream task.
5. Frontend resumes recognition for the next turn.

## Important Technical Notes

### Current strengths

- clean separation between orchestration, retrieval, and search
- low-latency token streaming path
- practical Qdrant + HNSW retrieval design
- user-filtered vector search
- voice-first UX instead of text-first chat with optional mic support
- interruptible TTS loop with barge-in detection

### Current limitations in the present code

- `DEFAULT_USER_ID` is effectively used for retrieval, so the current runtime is not fully multi-user yet
- the LangGraph config uses a fixed `thread_id` of `"5"`, so conversation isolation is not production-grade yet
- Redis exists in the repo but is not part of the active request path
- conversation state is in-memory only
- retrieval currently returns top 3 chunks without reranking
- document support is PDF-only
- upload persistence and cleanup strategy are minimal
- no authentication or tenant isolation layer exists yet

These do not remove the architectural value of the project, but they are worth documenting honestly if this README is used for demos, interviews, or portfolio sharing.

## Why This Project Is Technically Interesting

This project is more than a basic RAG demo because it combines several layers that are usually built separately:

- private document retrieval
- web-grounded research
- a routing agent that decides between them
- streamed answer delivery
- browser-native speech input/output
- interrupt-driven voice interaction

That combination makes it a strong example of a modern conversational systems project, especially for showcasing:

- retrieval architecture
- vector search design
- multi-agent orchestration
- real-time interaction systems
- voice UX engineering

## Future Improvement Ideas

- replace the fixed `user_id` and `thread_id` with session-scoped values
- add persistent chat memory
- add citations for retrieved chunks and search results
- add reranking before final RAG answer generation
- support more file types beyond PDFs
- move from browser STT/TTS to dedicated production speech services if needed
- introduce auth and true multi-tenant isolation
- add observability for latency, retrieval hits, and agent routing decisions

## Summary

`Voice With RAG` is a voice-first multi-agent retrieval system that combines:

- FastAPI for backend transport
- LangChain/LangGraph-style orchestration for routing
- Qdrant + HNSW for private semantic retrieval
- Ollama `bge-m3` embeddings
- Tavily for web search
- Next.js for a streaming speech interface with barge-in

It is a solid foundation for an AI assistant that can bridge private knowledge, public knowledge, and live voice interaction in a single application.
