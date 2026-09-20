#!/usr/bin/env bash

set -euo pipefail

BIN_DIR="${HOME}/.local/bin"
mkdir -p "$BIN_DIR"

command -v node >/dev/null || { printf 'node is required\n' >&2; exit 1; }
command -v python3 >/dev/null || { printf 'python3 is required\n' >&2; exit 1; }

# Standalone launchers for the MCP servers.
NODE_ROOT="${HOME}/.local/share/skynet-mcp-node"
npm install --prefix "$NODE_ROOT" @playwright/mcp@0.0.79
NODE_BIN="$(command -v node)"
printf '%s\n' '#!/bin/sh' "exec \"$NODE_BIN\" \"$NODE_ROOT/node_modules/@playwright/mcp/cli.js\" --headless --executable-path /usr/bin/chromium \"\$@\"" >"$BIN_DIR/playwright-mcp"
chmod 755 "$BIN_DIR/playwright-mcp"

PY_ROOT="${HOME}/.local/share/skynet-mcp-venv"
python3 -m venv "$PY_ROOT"
"$PY_ROOT/bin/python" -m pip install --upgrade arxiv-mcp-server==0.6.3 duckduckgo-mcp-server==0.6.1
printf '%s\n' '#!/bin/sh' "exec \"$PY_ROOT/bin/python\" -c 'import sys; from arxiv_mcp_server import main; sys.exit(main())' \"\$@\"" >"$BIN_DIR/arxiv-mcp-server"
printf '%s\n' '#!/bin/sh' "exec \"$PY_ROOT/bin/python\" -c 'import sys; from duckduckgo_mcp_server.server import main; sys.exit(main())' \"\$@\"" >"$BIN_DIR/duckduckgo-mcp-server"
chmod 755 "$BIN_DIR/arxiv-mcp-server" "$BIN_DIR/duckduckgo-mcp-server"

if [[ -x "${HOME}/.local/bin/github-mcp-server" ]]; then
    chmod 755 "${HOME}/.local/bin/github-mcp-server"
fi

printf 'MCP runtime installed:\n'
command -v playwright-mcp
command -v arxiv-mcp-server
command -v duckduckgo-mcp-server
if command -v github-mcp-server >/dev/null 2>&1; then
    command -v github-mcp-server
fi
