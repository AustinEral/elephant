# Use Elephant with Codex

Elephant does not need a plugin to work with Codex. The integration point is the MCP server at `/mcp`.

There are two good setup paths:

- repo setup: best when you are working inside this repository
- global setup: best when you are only running the Elephant server, including Docker-only installs

## What Codex needs

Codex needs two things:

- an MCP server configuration so it can reach Elephant
- repository instructions so it knows when to use Elephant

This repository provides both:

- [`.codex/config.toml`](../.codex/config.toml): project-scoped MCP configuration for trusted projects
- [`AGENTS.md`](../AGENTS.md): instructions that tell Codex when to use Elephant memory
- `.elephant/bank_id`: an optional untracked local file for pinning a default bank per repo

## Option 1: Repo Setup

Use this when you cloned the Elephant repository and launch Codex inside it.

1. Start Elephant:

```bash
docker compose up -d --build
```

2. Trust the repository in Codex.

3. Launch Codex from the repository root.

When the project is trusted, Codex can read [`.codex/config.toml`](../.codex/config.toml) and discover the local Elephant MCP server automatically. [`AGENTS.md`](../AGENTS.md) then helps Codex choose Elephant for memory tasks without repeated prompting.

If you want Codex to keep using the same bank for this repository without repeating the bank ID, create a local untracked file:

```bash
mkdir -p .elephant
echo "YOUR_BANK_ID_HERE" > .elephant/bank_id
```

That file is ignored by git and can safely stay local to your machine.

## Option 2: Global Setup

Use this when you are running Elephant outside this repository, including a Docker-only deployment.

Start Elephant first. For the default local server:

```bash
docker compose up -d --build
```

Then register the MCP server with Codex:

```bash
codex mcp add elephant --url http://127.0.0.1:3001/mcp
codex mcp list
```

You can also use the helper script from this repository:

```bash
./scripts/install-codex-mcp.sh
```

For a non-default URL:

```bash
./scripts/install-codex-mcp.sh http://host.docker.internal:3001/mcp
```

After global setup, add an `AGENTS.md` file to the repository where you want Codex to use Elephant, or explicitly tell Codex to use the `elephant` MCP server.

For a stable default bank in a working repository, you can use the same local `.elephant/bank_id` convention there too.

## Docker-Only Users

If you only distribute the Docker image or container stack, Codex will not auto-discover Elephant by itself. You still need one Codex-side setup step:

```bash
codex mcp add elephant --url http://127.0.0.1:3001/mcp
```

That is the main difference between:

- running Elephant
- teaching Codex where Elephant is

The Docker image solves the first problem. The Codex MCP config solves the second.

## Default Bank Recommendation

Do not use a server-global default bank right now.

Elephant does not currently include built-in auth or tenant isolation, so a server-wide “current bank” would be shared by all clients and agents using that server.

The safer pattern is:

- keep the server stateless about “current bank”
- pin a default bank locally per repository with `.elephant/bank_id`
- override it explicitly when you want a different bank

## Recommended AGENTS.md Snippet

If you want Elephant in another repository, add something like:

```md
Always use the `elephant` MCP server when it is available and the task would benefit from persistent memory about users, projects, procedures, or prior sessions.
```

## Troubleshooting

If Codex does not appear to use Elephant:

- confirm the server is up with `curl http://127.0.0.1:3001/v1/info`
- confirm Codex sees the MCP server with `codex mcp list`
- if using project config, make sure the repository is trusted
- if using another repository, add an `AGENTS.md` hint or explicitly tell Codex to use `elephant`

## Related Docs

- [Getting Started](getting-started.md)
- [API Reference](api.md)
