# Deployment

This document covers the deployment modes described in this repository and the operational assumptions behind each one.

## Deployment Modes

### 1. Local / dev

Supported and recommended for:

- evaluating Elephant
- local product development
- benchmark work
- small internal demos

This is the most exercised path in the repository.

### 2. Self-hosted single-node

Supported for technically capable teams that want to run Elephant themselves.

This means:

- one Elephant server process
- one PostgreSQL + pgvector database
- provider-backed LLM access
- persistent disk for database state

This is the simplest production-oriented path described here.

### 3. Managed / hosted Elephant

This repository does not provide a managed Elephant service.

### 4. Multi-node / HA / internet-wide public API

Elephant can sit behind standard infrastructure, but this repository does not include a complete opinionated deployment package for:

- autoscaling
- rolling upgrades across multiple app nodes
- built-in auth and tenant isolation
- public internet exposure without a separate gateway layer

## Runtime Requirements

Elephant needs:

- PostgreSQL with `pgvector`
- outbound network access to your configured LLM provider
- configured embedding and reranker backends
- a valid `.env` or equivalent environment configuration

Default ports:

- Elephant API + MCP: `3001`
- local Postgres in the provided Compose stack: `5434`

The server auto-runs database migrations on startup.

## Security Reality

Elephant does **not** ship with built-in API auth, and the server uses permissive CORS by default.

That means:

- do not expose Elephant directly to the public internet as-is
- put it behind your own auth layer, reverse proxy, and network controls
- treat it as a trusted internal service unless you add your own gateway protections

Minimum recommendation for any non-local environment:

- private network or VPN access
- TLS termination at a reverse proxy
- upstream auth at the proxy or platform layer
- database credentials managed as secrets, not hardcoded files

## Option 1: Docker Compose

This is the fastest path to a working self-hosted Elephant instance.

### What it starts

- `postgres` using `pgvector/pgvector:pg16`
- `elephant` built from the local repo

The Compose setup:

- exposes Postgres on host port `5434`
- exposes Elephant on host port `3001`
- mounts a named Docker volume for Postgres persistence
- sets container-safe model paths for ONNX runtime and local models

### Start it

```bash
cp .env.example .env
docker compose up -d --build
```

### Verify it

```bash
curl localhost:3001/v1/info
```

If that returns model information, the server is up and able to answer REST requests.

### Persistence

Postgres data is stored in the named Docker volume:

- `pgdata`

Removing containers without removing volumes preserves database state. Removing the volume deletes stored memory.

### When Compose is the right choice

Use this when you want:

- one-box local or team dev
- a quick internal demo
- a simple self-hosted deployment without orchestrators

## Option 2: Self-hosted from source

Use this when you want tighter control over the runtime, custom process management, or non-Docker deployment.

### Requirements

- Rust toolchain
- PostgreSQL with `pgvector`
- ONNX Runtime shared library if using local embeddings
- local embedding / reranker model files if using local ONNX models

### Start the database

Point `DATABASE_URL` at a Postgres instance with `pgvector` enabled.

Example:

```env
DATABASE_URL=postgres://elephant:elephant@localhost:5434/elephant
LISTEN_ADDR=0.0.0.0:3001
SERVER_AUTO_CONSOLIDATION=1
SERVER_AUTO_CONSOLIDATION_MIN_FACTS=32
SERVER_AUTO_CONSOLIDATION_COOLDOWN_SECS=30
SERVER_AUTO_CONSOLIDATION_MERGE_OPINIONS=0
```

### Configure providers

At minimum, configure:

- `LLM_PROVIDER`
- provider credentials
- retain / reflect model selection
- embedding backend
- reranker backend

Start from [.env.example](../.env.example) and fill in only the provider path you actually intend to use.

### Run the server

```bash
cargo run --release
```

The server binds to `LISTEN_ADDR` and auto-runs database migrations at startup.

By default, the HTTP/MCP server also evaluates background consolidation after successful `retain` calls. That maintenance is non-blocking and per-bank rate-limited. If you want fully manual maintenance instead, set `SERVER_AUTO_CONSOLIDATION=0`.

### When source deployment is the right choice

Use this when you want:

- systemd / Nomad / Kubernetes-style process control
- direct control over environment injection
- custom Postgres topology
- custom reverse proxy / ingress setup

## Configuration Checklist

Before calling a deployment ready, verify:

- `DATABASE_URL` points at the intended Postgres instance
- the database user can create tables and extensions needed by startup migrations
- LLM credentials are present and valid
- embedding configuration matches your bank creation strategy
- reranker configuration matches your deployment choice
- `LISTEN_ADDR` is set correctly for the environment
- server-side consolidation policy matches the deployment:
  - `SERVER_AUTO_CONSOLIDATION`
  - `SERVER_AUTO_CONSOLIDATION_MIN_FACTS`
  - `SERVER_AUTO_CONSOLIDATION_COOLDOWN_SECS`
  - `SERVER_AUTO_CONSOLIDATION_MERGE_OPINIONS`
- model paths are correct if using local embeddings or local reranking

## Health Checks

Use:

```bash
curl localhost:3001/v1/info
```

That confirms:

- the process is listening
- routing is active
- runtime configuration loaded successfully

For deeper functional checks, run the quickstart bank/create/retain/reflect flow from [getting-started.md](getting-started.md).

## Backups and Data Safety

Elephant stores memory in PostgreSQL. Treat Postgres backups as your source of truth.

At minimum:

- back up the Postgres database
- test restore procedures
- treat bank data as application data, not cache

If you use local model files outside Docker, manage those separately from database backups.

## Upgrades

Recommended upgrade flow:

- review release notes / code changes
- deploy new app version
- allow Elephant startup to run migrations
- verify `/v1/info`
- run a small retain/reflect smoke test

For cautious environments, use a staging database and restore path before production upgrades.

## Not Provided By This Repository

These areas are not provided as polished built-in product features:

- built-in auth and tenant isolation
- turnkey multi-node deployment
- managed hosting
- opinionated observability stack
- one-click cloud deployment templates

If you are comfortable wiring those pieces yourself, Elephant is usable as infrastructure. If you need them provided out of the box, add them at your platform layer or wait for explicit support to be documented.

## Related Docs

- [getting-started.md](getting-started.md)
- [api.md](api.md)
- [architecture.md](architecture.md)
