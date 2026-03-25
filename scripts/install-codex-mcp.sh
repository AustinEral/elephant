#!/usr/bin/env bash

set -euo pipefail

SERVER_URL="${1:-${ELEPHANT_MCP_URL:-http://127.0.0.1:3001/mcp}}"

if ! command -v codex >/dev/null 2>&1; then
  echo "error: codex CLI is not installed or not on PATH" >&2
  exit 1
fi

echo "Adding Elephant MCP server to Codex: ${SERVER_URL}"
codex mcp add elephant --url "${SERVER_URL}"

echo
echo "Configured MCP servers:"
codex mcp list
