#!/usr/bin/env bash
# Build pre-purged tailwind.css из templates агента.
# Standalone CLI без node_modules: один binary, на Linux/macOS/Windows.
#
# Usage:
#   ./webui/tailwind/build.sh          # production build (minified)
#   ./webui/tailwind/build.sh --watch  # dev watch mode
#
# Output: src/harnes/webui/static/css/tailwind.css
# При наличии этого файла base.html переключается с CDN на pre-built.

set -euo pipefail

VERSION="v3.4.17"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BIN_DIR="$SCRIPT_DIR/bin"
OUT_CSS="$ROOT_DIR/src/harnes/webui/static/css/tailwind.css"
INPUT_CSS="$SCRIPT_DIR/input.css"
CONFIG="$SCRIPT_DIR/tailwind.config.js"

uname_s=$(uname -s)
uname_m=$(uname -m)

case "$uname_s-$uname_m" in
    Linux-x86_64)   BIN_NAME="tailwindcss-linux-x64"   ;;
    Linux-aarch64)  BIN_NAME="tailwindcss-linux-arm64" ;;
    Darwin-x86_64)  BIN_NAME="tailwindcss-macos-x64"   ;;
    Darwin-arm64)   BIN_NAME="tailwindcss-macos-arm64" ;;
    *) echo "Unsupported platform: $uname_s-$uname_m" >&2; exit 1 ;;
esac

BIN_PATH="$BIN_DIR/$BIN_NAME"
URL="https://github.com/tailwindlabs/tailwindcss/releases/download/${VERSION}/${BIN_NAME}"

if [ ! -x "$BIN_PATH" ]; then
    echo "→ downloading $BIN_NAME ($VERSION)..."
    mkdir -p "$BIN_DIR"
    curl -sSL --fail -o "$BIN_PATH" "$URL"
    chmod +x "$BIN_PATH"
fi

mkdir -p "$(dirname "$OUT_CSS")"

EXTRA_FLAGS=("--minify")
if [ "${1:-}" = "--watch" ]; then
    EXTRA_FLAGS=("--watch")
fi

echo "→ building tailwind.css (config: $CONFIG)"
cd "$SCRIPT_DIR"
"$BIN_PATH" \
    -i "$INPUT_CSS" \
    -c "$CONFIG" \
    -o "$OUT_CSS" \
    "${EXTRA_FLAGS[@]}"

if [ "${1:-}" != "--watch" ]; then
    size=$(wc -c < "$OUT_CSS")
    echo "✓ wrote $OUT_CSS (${size} bytes)"
fi
