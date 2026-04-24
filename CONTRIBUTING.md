# Contributing

Thank you for investing time in this project. Contributions are welcome through issues and pull requests.

## Before you start

- Review [README.md](README.md) for scope, requirements, and usage.
- Search [existing issues](https://github.com/craigh33/adk-go-bedrock/issues) and pull requests for duplicates.
- Prefer small, focused changes—one logical concern per PR keeps review straightforward.

## Development setup

- **Go**: match `go.mod` (see [`go.mod`](go.mod); currently Go 1.25+).
- **Clone** the repository and work on a branch off `main`.

### Homebrew Bundle (macOS)

The repository includes a [`Brewfile`](Brewfile) for [Homebrew Bundle](https://docs.brew.sh/Manpage#bundle-subcommand). From the repository root:

```bash
brew bundle install
```

That installs **`make`**, **`go`**, **`pre-commit`**, **`gitleaks`**, **`golangci-lint`**, and **`goreleaser`**. For typical contributions you only need the first five; **`goreleaser`** is for maintainers releasing binaries.

If you do not use macOS or Homebrew, install the equivalent tools yourself (see [Pre-commit](#pre-commit-required) for what must be on your `PATH` for hooks versus `make lint`).

### Makefile

The root [`Makefile`](Makefile) defines these targets:

| Target | Description |
|--------|-------------|
| `make test` | Run unit tests (`go test ./... -count=1`) |
| `make build` | Compile all packages (`go build ./...`) |
| `make lint` | Run `golangci-lint run ./...` (see [.golangci.yaml](.golangci.yaml)) |
| `make pre-commit-install` | Install pre-commit and `commit-msg` hooks (delegates to `make pre-commit`; if `pre-commit` is missing, the Makefile runs `brew install pre-commit`. Without Homebrew, install `pre-commit` yourself first—see [Pre-commit](#pre-commit-required)) |

`make pre-commit-install` is the same as `make pre-commit` for hook installation; the only difference is naming. If `pre-commit` is not on your `PATH`, the Makefile attempts `brew install pre-commit`, which requires Homebrew. On systems without Homebrew, install `pre-commit` manually (for example from [pre-commit.com](https://pre-commit.com/#install)) before running either target.

Run checks locally before pushing:

```bash
make test
make lint
make build
```

## Pre-commit (required)

Contributions **must** pass [pre-commit](https://pre-commit.com) checks. Run `pre-commit run --all-files` locally before opening or updating a PR. On pull requests, CI currently runs `golangci-lint` and `go test ./... -count=1` as defined in [`.github/workflows/ci-build.yaml`](.github/workflows/ci-build.yaml).

The `no-commit-to-branch` hook blocks commits when your checked-out branch is `main`; use a feature branch instead.

If you already ran **`brew bundle install`** on macOS, `pre-commit`, `golangci-lint`, and `gitleaks` are on your `PATH`, which covers hooks and **`make lint`** in one step.

**Pre-commit hooks** need **`pre-commit`** on your `PATH`, plus **`gitleaks`**: the [gitleaks hook](.pre-commit-config.yaml) is a local `language: system` hook, so it does not bundle a binary. The **`golangci-lint`** hook uses the [golangci-lint pre-commit repo](https://github.com/golangci/golangci-lint), which installs a pinned linter for you—you do **not** need a separate system `golangci-lint` install for `pre-commit run` or git hooks.

**`make lint`** runs `golangci-lint` from your shell, so that target **does** need **`golangci-lint`** installed and on your `PATH` (unless you rely on the hook or CI only).

Manual installs (when not using Homebrew Bundle):

| Tool | Needed for | Install (examples) |
|------|------------|-------------------|
| [pre-commit](https://pre-commit.com) | Hooks | `brew install pre-commit` or [other install methods](https://pre-commit.com/#install) |
| [gitleaks](https://github.com/gitleaks/gitleaks) | Hooks (local hook) | `brew install gitleaks` or [release binaries](https://github.com/gitleaks/gitleaks/releases) |
| [golangci-lint](https://golangci-lint.run/welcome/install/) | `make lint` only (not required for the pre-commit golangci-lint hook) | `brew install golangci-lint` or upstream install docs |

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
