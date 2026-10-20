# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run the server (from project root)
./run.sh

# Run manually
cd backend && uv run uvicorn app:app --reload --port 8000

# Lint (run after major edits)
uv run npm run lint
```

The app serves at `http://localhost:8000`. Requires `ANTHROPIC_API_KEY` in a `.env` file at the project root.

## Architecture

Full-stack RAG chatbot. The FastAPI backend serves the static frontend from `frontend/` and exposes two API endpoints: `POST /api/query` and `GET /api/courses`.

**Query flow:**
1. Frontend (`frontend/script.js`) POSTs `{ query, session_id }` to `/api/query`
2. `app.py` delegates to `RAGSystem.query()`
3. `RAGSystem` sends the query to `AIGenerator` along with Claude's tool definitions and conversation history
4. Claude decides whether to call `search_course_content` (a tool backed by `VectorStore`)
5. If Claude calls the tool, `VectorStore.search()` queries ChromaDB, results go back to Claude, and Claude makes a second API call to synthesize the final answer
6. Sources are tracked on `CourseSearchTool.last_sources` and returned alongside the response

**Component responsibilities:**
- `backend/rag_system.py` — top-level orchestrator; wires all components together
- `backend/ai_generator.py` — Anthropic SDK wrapper; handles the tool-use loop (two-call pattern: first call may return `tool_use`, second call synthesizes after tool results)
- `backend/vector_store.py` — ChromaDB wrapper with two collections: `course_catalog` (course metadata, used for fuzzy course name resolution) and `course_content` (chunked text, used for semantic search)
- `backend/document_processor.py` — parses structured `.txt` course files into `Course`/`Lesson`/`CourseChunk` objects and splits lesson text into sentence-based chunks
- `backend/search_tools.py` — defines the `search_course_content` Anthropic tool and `ToolManager` registry
- `backend/session_manager.py` — in-memory conversation history, rolling window of `MAX_HISTORY=2` exchanges

**Course document format** (files in `docs/`):
```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>
Lesson 1: <title>
Lesson Link: <url>
<lesson content>
Lesson 2: ...
```

Documents are loaded from `docs/` on startup; already-indexed courses (matched by title) are skipped. ChromaDB is persisted at `backend/chroma_db/`.

**Config** (`backend/config.py`): all tunables live here — model, chunk size/overlap, max search results, history window, ChromaDB path. Model is `claude-sonnet-4-20250514`. Embeddings use `all-MiniLM-L6-v2` via `sentence-transformers`.
