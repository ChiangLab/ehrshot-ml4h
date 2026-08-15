#!/usr/bin/env bash

set -euo pipefail

REMOTE_URL="${1:?Usage: configure_mednet_remote.sh <exact-repository-url> [verified-email]}"
VERIFIED_EMAIL="${2:-${MEDNET_GIT_EMAIL:-}}"
if [[ -z "${VERIFIED_EMAIL}" ]]; then
  echo "Provide the email verified by the MedNet GitHub service as argument 2 or MEDNET_GIT_EMAIL." >&2
  exit 1
fi

git config --local user.name "William Delery"
git config --local user.email "${VERIFIED_EMAIL}"
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "${REMOTE_URL}"
else
  git remote add origin "${REMOTE_URL}"
fi

bash scripts/git/pre_push_safety.sh
echo "Configured local author and origin for account wdelery."
echo "The numeric account ID is not an authentication credential and was not stored."
