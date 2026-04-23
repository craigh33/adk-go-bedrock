# Contributing

Thank you for investing time in this project. Contributions are welcome through issues and pull requests.

## Fork-based workflow

This repository uses a **fork-based** contribution model: changes land via pull requests from forks, not from new branches pushed directly to this repo (unless you are a maintainer with explicit access).

1. **Fork** this repository on GitHub (your account owns the fork).
2. **Clone your fork** and add this repo as `upstream`:

   ```bash
   git clone https://github.com/YOUR_GITHUB_USERNAME/adk-go-bedrock.git
   cd adk-go-bedrock
   git remote add upstream https://github.com/craigh33/adk-go-bedrock.git
   ```

3. **Sync before you branch** (optional but recommended):

   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

4. **Create a branch** from `main` for your change:

   ```bash
   git checkout -b feat/your-change
   ```

5. **Commit, push to your fork**, and open a **pull request** from your branch into `craigh33/adk-go-bedrock:main`.

Maintainers cannot push fixes to your PR branch unless you grant [maintainer edits](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/allowing-changes-to-a-pull-request-branch-created-from-a-fork); enabling that speeds up small follow-ups.

## Before you start

- Review [README.md](README.md) for scope, requirements, and usage.
- Search [existing issues](https://github.com/craigh33/adk-go-bedrock/issues) and pull requests for duplicates.
- Prefer small, focused changes—one logical concern per PR keeps review straightforward.

## Development setup

- **Go**: match `go.mod` (see [`go.mod`](go.mod); currently Go 1.25+).
- **Clone your fork** and work on a branch off `main` (see [Fork-based workflow](#fork-based-workflow)).

### Homebrew Bundle (macOS)

The repository includes a [`Brewfile`](Brewfile) for [Homebrew Bundle](https://docs.brew.sh/Manpage#bundle-subcommand). From the repository root:

```bash
brew bundle install
```

That installs **`make`**, **`go`**, **`pre-commit`**, **`gitleaks`**, **`golangci-lint`**, and **`goreleaser`**. For typical contributions you only need the first five; **`goreleaser`** is for maintainers releasing binaries.

If you do not use macOS or Homebrew, install the equivalent tools yourself (see [Pre-commit](#pre-commit-required) for what pre-commit expects on your `PATH`).

### Makefile

The root [`Makefile`](Makefile) defines these targets:

| Target | Description |
|--------|-------------|
| `make test` | Run unit tests (`go test ./... -count=1`) |
| `make build` | Compile all packages (`go build ./...`) |
| `make lint` | Run `golangci-lint run ./...` (see [.golangci.yaml](.golangci.yaml)) |
| `make pre-commit-install` | Install pre-commit and `commit-msg` hooks (requires `pre-commit` on your `PATH`; see [Pre-commit](#pre-commit-required)) |

`make pre-commit-install` delegates to `make pre-commit`, which may run `brew install pre-commit` on macOS if the binary is missing.

Run checks locally before pushing:

```bash
make test
make lint
make build
```

## Pre-commit (required)

Contributions **must** pass [pre-commit](https://pre-commit.com) checks. On pull requests, CI runs `pre-commit run --all-files` then `go test ./... -count=1` ([`.github/workflows/ci-build.yaml`](.github/workflows/ci-build.yaml)); run the same locally before opening or updating a PR.

The `no-commit-to-branch` hook blocks commits when your checked-out branch is `main`. Use a feature branch (see [Fork-based workflow](#fork-based-workflow)).

If you already ran **`brew bundle install`** on macOS, `pre-commit`, `golangci-lint`, and `gitleaks` are on your `PATH`. Otherwise install these tools manually:

| Tool | Purpose | Install |
|------|---------|---------|
| [pre-commit](https://pre-commit.com) | Hook framework | `brew install pre-commit` |
| [golangci-lint](https://golangci-lint.run/welcome/install/) | Go linter (`make lint` mirrors the hook) | `brew install golangci-lint` |
| [gitleaks](https://github.com/gitleaks/gitleaks) | Secret / credential scanner (also invoked by pre-commit) | `brew install gitleaks` |

Wire hooks into your clone:

```bash
make pre-commit-install
```

This installs hooks for both the `pre-commit` and `commit-msg` stages.

| Hook | Stage | Description |
|------|-------|-------------|
| `trailing-whitespace` | pre-commit | Strips trailing whitespace |
| `end-of-file-fixer` | pre-commit | Ensures files end with a newline |
| `check-yaml` | pre-commit | Validates YAML syntax |
| `no-commit-to-branch` | pre-commit | Prevents direct commits to `main` |
| `conventional-pre-commit` | commit-msg | Enforces [Conventional Commits](https://www.conventionalcommits.org/) (`feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`) |
| `golangci-lint` | pre-commit | Runs configured lint on Go files (aligned with `make lint`) |
| `gitleaks` | pre-commit | Scans for secrets |

To match CI before you push:

```bash
pre-commit run --show-diff-on-failure --color always --all-files
```

If a hook fails, fix the findings or let the hook auto-fix what it can, then commit again.

## Pull requests

- Open PRs **from your fork** into `main` on this repository (see [Fork-based workflow](#fork-based-workflow)).
- Fill out the PR template and describe **what** changed and **why** (not only the diff).
- Add or update tests when behavior changes.
- Update user-facing docs (for example `README.md` or example READMEs) when behavior or setup changes.
- Keep commits reasonably clean; maintainers may squash on merge when that helps history.

## Security

If you find a security vulnerability, please **do not** open a public issue. Use [GitHub Security Advisories](https://github.com/craigh33/adk-go-bedrock/security/advisories) for this repository if available, or contact the maintainers privately with enough detail to reproduce and assess impact.

## Code of conduct

Participation is expected to be respectful and professional. Harassment or abuse is not tolerated.

## License

By contributing, you agree that your contributions are licensed under the same terms as the project ([Apache License 2.0](LICENSE)).
