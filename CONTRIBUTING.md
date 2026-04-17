# Contributing to RAG-Lab

Thanks for your interest in contributing! This document covers the basics for getting set up and the security expectations for an open-source project.

## Development setup

```bash
# Backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Frontend
cd frontend
npm install

# Pre-commit hooks (recommended)
pip install pre-commit
pre-commit install
```

The pre-commit hooks run [gitleaks](https://github.com/gitleaks/gitleaks) to catch accidentally committed secrets, and a few standard checks (large files, private keys, YAML/JSON validity).

## Security expectations

This is an open-source project — public history is forever. Please follow these rules:

1. **Never commit `.env` files.** Only `.env.example` is tracked, and it must contain placeholder values only (never realistic-looking fakes).
2. **Never commit credentials, API keys, tokens, or private keys.** Use environment variables.
3. **Never commit user data.** SQLite databases, session JSON, uploaded documents, vector indexes, and conversation memory are gitignored — keep it that way.
4. **Run `pre-commit install`** before your first commit. This adds gitleaks as a guard.
5. **For production deployments**, set `JWT_SECRET` explicitly in your `.env`. The dev-mode auto-generated secret invalidates all sessions on restart and is unsafe for production.

If you accidentally commit a secret:
- Rotate the credential immediately at the provider.
- Force-push a history rewrite (`git filter-repo` or BFG) and notify maintainers — `git rm` alone leaves the secret in history.

## Submitting changes

- Open an issue first for non-trivial changes so we can discuss approach.
- Keep PRs focused — one logical change per PR.
- Include a clear description of what the change does and why.
