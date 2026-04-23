## Summary

<!-- Short explanation of the problem or motivation and what this PR changes. -->

This project uses a **fork-based** workflow: PRs should come from a **branch on your fork**, not a branch on this repo (unless you are a maintainer). See [CONTRIBUTING.md](CONTRIBUTING.md).

## Type of change

<!-- Mark relevant items with an x like [x]. -->

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature / enhancement (non-breaking change that adds behavior)
- [ ] Breaking change (fix or feature that could break existing callers)
- [ ] Documentation only
- [ ] Build / CI / tooling
- [ ] Refactor / test / chore (no user-visible behavior change)

## Testing

<!-- How did you verify this? Commands run, manual scenarios, etc. -->

- [ ] `make test`
- [ ] `make lint` (or equivalent local golangci-lint)
- [ ] Using an example (if a new one is created, include it in the summary)

## Checklist

- [ ] This PR is from **my fork** (fork-based contributions).
- [ ] I ran **`pre-commit run --all-files`** (or equivalent local hooks) and CI will pass pre-commit.
- [ ] My change follows existing patterns in this repository (naming, layout, error handling).
- [ ] I updated docs when behavior, setup, or public API surface changed (if applicable).
- [ ] I considered backwards compatibility for library consumers (if applicable).

## Related issues

<!-- Links to issues this closes or relates to, e.g. Fixes #123 -->
