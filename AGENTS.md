# AGENTS.md - Agent Guide for Embedding Query

This file gives you the agentic context you need to work on this codebase safely. For project overview, features, install / deploy steps and human-facing security / performance / troubleshooting material, read @README.md — that file is canonical and not duplicated here.

This is the single canonical guide for autonomous coding agents in this repository. Assistant-specific files such as @CLAUDE.md, and any future tool-specific guide such as `GEMINI.md`, should only point here and should not duplicate repository instructions.

Deeper specs live in their own files:

- For any project update, keep documentation aligned:
  - Update `README.md` for user-facing behavior, configuration, setup, deployment, troubleshooting, or verification changes.
  - Update this file only when agent workflow or safety context changes.

---

## Related repositories (project ecosystem)

`embedding-query` is one stage of **Agent BBB**, a multi-repository movie/TV database system owned by GitHub user `vaugouin`. All sibling repos live under `%USERPROFILE%/Code/<repo>` and at `github.com/vaugouin/<repo>`; they are interdependent stages of one pipeline that converges on a shared MySQL/MariaDB database (`T_WC_*` tables) and a ChromaDB vector store. The canonical roster of sibling repositories is kept in `doc/related-repositories/related-repositories.txt` in the `tmdb-front` repo.

Pipeline stages:
- **Infrastructure** — `python` (shared crawler base image), `chromadb` (vector service), `reverseproxy` (NGINX TLS ingress), `chromadb-security-test` (firewall validation).
- **Acquisition** — `tmdb-crawler`, `imdb-crawler`, `sparql-crawler`, `sparql-movies-persons`, `wikidata-crawler`, `wikipedia-crawler`, `selenium-tmdb`, `download-images`, `sqlite-plex-to-tmdb`, `movieparadise`.
- **Preprocessing → `T_WC_T2S_*`** — `tmdb-movie-preprocess`, `tmdb-person-preprocess`, `keywords-processing`.
- **Semantic index & name resolution** — `embedding-update`, `embedding-query`, `rapidfuzz_query`.
- **Serving** — `fastapi-text2sql` (NL→SQL API + MCP server), `voice-agent`, `tmdb-front` (PHP web front-end).
- **Evaluation** — `eval-text2sql`, `extract-movie-questions`.
- **Maintenance & tooling** — `plex-duplicates`, `subtitle-translate`, `powershell`, `playwright-test`.
- **Monitoring & observability** — `data-monitoring`.

**This repository's role:** Semantic-index tooling. Interactive CLI that runs similarity searches against the ChromaDB collections maintained by `embedding-update` (via the `chromadb` service) — used to test and validate the same semantic search that `fastapi-text2sql` performs at query time.

---

## Where things live (file → role)

Edit at the right layer; the architecture is intentionally split.

## Code conventions

- **Hungarian notation** for variables (legacy style):
  - `str` — strings (`strtablename`, `strapiversion`)
  - `lng` — integers (`lngpage`, `lngrowsperpage`)
  - `dbl` — floats (`dblavailableram`)
  - `arr` — lists / arrays
  - `int` — boolean-like flags (`intcleanupenabled`, `intentity`)
- **Function naming**: public pipeline entry points use `f_` (`f_text2sql`, `f_entity_extraction`, `f_resolve_complex_question`, `f_answer_single_value`, `f_hello_world`); private helpers use `_` (`_call_chat_llm`, `_normalize_llm_model`).
- **Docstrings**: Google-style on public functions.
- **Error handling**: broad try/except with console logging; surface failures via the `error` response field and the `messages` trace. Database execution errors are not returned directly to clients — they go through the complex-question retry path when enabled.
- **JSON serialization**: use `logs.decimal_serializer()` for `Decimal` and `datetime`.

---

## Encoding

Keep Markdown, prompt files, JSON config, and logs UTF-8. These files contain non-ASCII names and multilingual examples. Avoid editor or terminal operations that rewrite them with mojibake.

---

## Build & deployment (Docker)

The query CLI is built and run as a Docker container via the repo's `Dockerfile` (base image `python:3.12-slim-bookworm`, `PYTHONUNBUFFERED=1`). It installs `requirements.txt`, copies the repo, and runs `CMD ["python", "embedding-query.py"]`. No ports or volumes are exposed; it reaches the `chromadb` service over the network using runtime configuration.

---

**Last Updated**: 2026-06-03
**Current Version**: 1.0.0 
