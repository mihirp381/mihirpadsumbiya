#!/usr/bin/env bash
# Builds the site with the GitHub Pages builder image and serves it locally.
set -euo pipefail

port="${1:-8731}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
name="$(basename "$repo")"
root="$(dirname "$repo")/_preview"
out="$root/$name"

mkdir -p "$out"
docker run --rm --platform linux/amd64 \
  -v "$repo":/src:ro -v "$out":/out \
  -e JEKYLL_ENV=production -e PAGES_REPO_NWO=mihirp381/mihirpadsumbiya \
  --entrypoint /bin/bash ghcr.io/actions/jekyll-build-pages:v1.0.13 \
  -c 'cd "$BUNDLE_APP_CONFIG" && bin/github-pages build -s /src -d /out'

echo "Serving http://127.0.0.1:$port/$name/ (Ctrl-C to stop)"
cd "$root" && exec python3 -m http.server "$port" --bind 127.0.0.1
