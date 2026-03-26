# Use Elephant with Codex

Run Elephant with Docker:

```bash
docker compose up -d --build
```

## In this repository

This repo already includes [`.codex/config.toml`](../.codex/config.toml), so Codex can discover Elephant automatically when you launch Codex from the repository root in a trusted project.

To pin a repo-local bank:

```bash
mkdir -p .elephant
echo "YOUR_BANK_ID_HERE" > .elephant/bank_id
```

## Outside this repository

Register Elephant once with Codex:

```bash
codex mcp add elephant --url http://127.0.0.1:3001/mcp
codex mcp list
```

To set a personal default bank for Codex:

```bash
mkdir -p ~/.codex/elephant
echo "YOUR_BANK_ID_HERE" > ~/.codex/elephant/bank_id
```

Then add this to `~/.codex/AGENTS.md`:

```md
If `.elephant/bank_id` exists, use the memory it selects unless the user explicitly asks for a different one.
Otherwise, if `~/.codex/elephant/bank_id` exists, use the memory it selects.
```

Project-local `.elephant/bank_id` still overrides the global Codex bank because project `AGENTS.md` files are more specific than the global Codex instructions.

## Verify

```bash
curl http://127.0.0.1:3001/v1/info
codex mcp list
```

## Related Docs

- [Use Elephant with Claude](claude.md)
- [Getting Started](getting-started.md)
- [API Reference](api.md)
