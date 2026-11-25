#!/usr/bin/env bash
# Deploy the current commit to a Hugging Face Space (Docker SDK).
#
#   HF_TOKEN=hf_... deploy/huggingface/deploy.sh <hf-username>/<space-name>
#
# The Space gets the repository at HEAD, with deploy/huggingface/Dockerfile and README.md (the Space
# card) at its root. Hugging Face then builds the image, which indexes the demo repositories.
set -euo pipefail

SPACE="${1:?usage: HF_TOKEN=... deploy/huggingface/deploy.sh <hf-username>/<space-name>}"
: "${HF_TOKEN:?set HF_TOKEN to a Hugging Face token with write access}"

root="$(git rev-parse --show-toplevel)"
rev="$(git -C "$root" rev-parse --short HEAD)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

git -C "$root" archive HEAD | tar -x -C "$tmp"
cp "$root/deploy/huggingface/Dockerfile" "$tmp/Dockerfile"
cp "$root/deploy/huggingface/README.md" "$tmp/README.md"

cd "$tmp"
git init -q -b main
git add -A
git -c user.name="BadeBhai deploy" -c user.email="deploy@users.noreply.huggingface.co" commit -q -m "Deploy BadeBhai $rev"
git push -q --force "https://user:${HF_TOKEN}@huggingface.co/spaces/${SPACE}" main
echo "Pushed $rev to https://huggingface.co/spaces/${SPACE} (the build takes about 10 minutes)."
