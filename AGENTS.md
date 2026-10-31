# AGENTS.md - Agent Guide for Embedding Query

Canonical agent guide for this repo (`CLAUDE.md` and any other tool-specific file only point here). Overview, commands, collections, configuration, Docker and troubleshooting are in @README.md. Update `README.md` for user-facing changes, this file only when agent workflow or safety context changes.

---

## Related repositories (project ecosystem)

Semantic-index tooling of **Agent BBB** (`github.com/vaugouin/<repo>`, cloned under `%USERPROFILE%/Code/<repo>`): an interactive CLI that runs similarity searches against the ChromaDB collections built by `embedding-update` (served by `chromadb`), to test the same semantic search `fastapi-text2sql` performs at query time. Roster: `%USERPROFILE%/Nestor/projets/t2s-backlog/topics/related-repositories.txt`; stage map: `tmdb-front/AGENTS.md` § Related repositories.

---

## Code conventions

- **Hungarian notation** for variables (legacy style): `str` strings, `lng` integers, `dbl` floats, `arr` lists, `int` boolean-like flags. Functions use the `f_` prefix.
- Keep Markdown, prompts, JSON config and logs UTF-8 (non-ASCII names); avoid tools that rewrite them with mojibake.

---

## Collections: what an agent must not undo

**⚠ `t2slocations` and `t2ssourceworks` are opened with `get_collection`, never `get_or_create_collection`.** Their HNSW configuration is set at creation by `embedding-update` (processes 216/217) and can never change, so whoever creates first decides it for every reader; a `get_or_create` here would silently recreate a missing collection with server defaults, and nothing would report it (searches just rank worse). Same rule in `fastapi-text2sql`. The legacy collections keep `get_or_create_collection` (created that way, on server defaults). When the collection is absent, startup says why and disables the command instead of creating it. Details: README § The two locations collections.

---

**Last Updated**: 2026-10-31
**Current Version**: 1.0.2 

## Backlog (Nestor second-brain)

Backlog (`EMBEDDING-QUERY-NNN`, status and priority) in the Nestor repo: `C:\Users\vaugo\Nestor\projets\t2s-backlog\repos\embedding-query.md` (dashboard: `...\t2s-backlog\index.md`). Consult it before implementing. Local paths on Philippe's PC only; they do not resolve on the VPS or cloud agents.
