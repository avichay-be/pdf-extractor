#!/usr/bin/env bash
set -euo pipefail

# Helper to push required secrets to GitHub using `gh` CLI.
# Requires: gh authenticated (gh auth login) and repo write access.

if ! command -v gh >/dev/null 2>&1; then
  echo "Error: gh CLI not found. Install and run 'gh auth login' first." >&2
  exit 2
fi

usage() {
  cat <<EOF
Usage: $0 --repo owner/repo \
  --azure-client-id <id> --azure-tenant-id <id> --azure-subscription-id <id> \
  --acr-password <password> [--mistral-key <key> --gemini-key <key> --api-key <key>]

This will set secrets in the given GitHub repository.
EOF
}

if [ "$#" -eq 0 ]; then
  usage
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO="$2"; shift 2;;
    --azure-client-id) AZURE_CLIENT_ID="$2"; shift 2;;
    --azure-tenant-id) AZURE_TENANT_ID="$2"; shift 2;;
    --azure-subscription-id) AZURE_SUBSCRIPTION_ID="$2"; shift 2;;
    --acr-password) ACR_PASSWORD="$2"; shift 2;;
    --mistral-key) AZURE_API_KEY="$2"; shift 2;;
    --gemini-key) GEMINI_API_KEY="$2"; shift 2;;
    --api-key) API_KEY="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [ -z "${REPO:-}" ] || [ -z "${AZURE_CLIENT_ID:-}" ] || [ -z "${AZURE_TENANT_ID:-}" ] || [ -z "${AZURE_SUBSCRIPTION_ID:-}" ] || [ -z "${ACR_PASSWORD:-}" ]; then
  echo "Missing required arguments" >&2
  usage
  exit 1
fi

echo "Setting secrets on $REPO..."

gh secret set AZURE_CLIENT_ID --repo "$REPO" --body "$AZURE_CLIENT_ID"
gh secret set AZURE_TENANT_ID --repo "$REPO" --body "$AZURE_TENANT_ID"
gh secret set AZURE_SUBSCRIPTION_ID --repo "$REPO" --body "$AZURE_SUBSCRIPTION_ID"
gh secret set ACR_PASSWORD --repo "$REPO" --body "$ACR_PASSWORD"

if [ -n "${AZURE_API_KEY:-}" ]; then
  gh secret set AZURE_API_KEY --repo "$REPO" --body "$AZURE_API_KEY"
fi
if [ -n "${GEMINI_API_KEY:-}" ]; then
  gh secret set GEMINI_API_KEY --repo "$REPO" --body "$GEMINI_API_KEY"
fi
if [ -n "${API_KEY:-}" ]; then
  gh secret set API_KEY --repo "$REPO" --body "$API_KEY"
fi

echo "Secrets set successfully. Verify in repo settings -> Secrets & variables -> Actions."
