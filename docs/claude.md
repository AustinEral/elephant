# Use Elephant with Claude

Run Elephant with Docker:

```bash
docker compose up -d --build
```

## In this repository

This repo already includes [`.mcp.json`](../.mcp.json), so Claude Code can discover Elephant automatically when you launch Claude Code from the repository root.

[`CLAUDE.md`](../CLAUDE.md) contains the Claude-specific Elephant usage guidance for this repository.

To pin a repo-local bank:

```bash
mkdir -p .elephant
echo "YOUR_BANK_ID_HERE" > .elephant/bank_id
```

## Outside this repository

Register Elephant once with Claude Code:

```bash
claude mcp add --transport http elephant http://127.0.0.1:3001/mcp
claude mcp list
```

To set a personal default bank for Claude Code:

```bash
mkdir -p ~/.claude/elephant
echo "YOUR_BANK_ID_HERE" > ~/.claude/elephant/bank_id
```

Then add this to `~/.claude/CLAUDE.md`:

```md
If `.elephant/bank_id` exists, use the memory it selects unless the user explicitly asks for a different one.
Otherwise, if `~/.claude/elephant/bank_id` exists, use the memory it selects.
```

Project-local `.elephant/bank_id` still overrides the global Claude bank because project `CLAUDE.md` files are more specific than the user-level Claude instructions.

## Verify

```bash
curl http://127.0.0.1:3001/v1/info
claude mcp list
```

## Related Docs

- [Use Elephant with Codex](codex.md)
- [Getting Started](getting-started.md)
- [API Reference](api.md)
