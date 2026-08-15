#!/usr/bin/env bash

set -euo pipefail

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Not inside a Git repository." >&2
  exit 1
fi

candidate_files="$(git ls-files --cached --others --exclude-standard)"
blocked_pattern='(^|/)(data/raw|data/derived|outputs/private|outputs/governed|models|model_cache|embeddings|patient_logs)(/|$)|\.(parquet|zip|duckdb|db|sqlite|safetensors|gguf|pt|pth)$'
candidate_blocked="$(printf '%s\n' "${candidate_files}" | rg -i "${blocked_pattern}" || true)"
if [[ -n "${candidate_blocked}" ]]; then
  echo "BLOCKED: governed data or model artifacts could be committed:" >&2
  echo "${candidate_blocked}" >&2
  exit 1
fi

large_candidates="$(git ls-files -z --cached --others --exclude-standard | xargs -0 -I{} sh -c 'test -f "$1" && test "$(wc -c < "$1")" -gt 50000000 && echo "$1"' sh {} || true)"
if [[ -n "${large_candidates}" ]]; then
  echo "BLOCKED: unignored files larger than 50 MB require manual review:" >&2
  echo "${large_candidates}" >&2
  exit 1
fi

secret_hits="$(git ls-files -z --cached --others --exclude-standard | xargs -0 rg -n -I \
  '(-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----|sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,})' || true)"
if [[ -n "${secret_hits}" ]]; then
  echo "BLOCKED: possible credential in a tracked file:" >&2
  echo "${secret_hits}" >&2
  exit 1
fi

echo "PASS: no tracked governed-data extensions, model weights, oversized files, or obvious credentials."
echo "Reminder: this check cannot prove that free text is deidentified; review every staged diff."
