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

If you do not use macOS or Homebrew, install the equivalent tools yourself (see [Pre-commit](#pre-commit-required)).

### Makefile

The root [`Makefile`](Makefile) defines these targets:

| Target | Description |
|--------|-------------|
| `make test` | Run unit tests (`go test ./... -count=1`) |
| `make build` | Compile all packages (`go build ./...`) |
| `make lint` | Run `golangci-lint run ./...` (see [.golangci.yaml](.golangci.yaml)) |
| `make pre-commit-install` | Install `pre-commit` and `commit-msg` hooks (same as `make pre-commit`; tries `brew install pre-commit` if the binary is missing) |

Before you push, run pre-commit plus the same Makefile targets CI uses ([workflow](.github/workflows/ci-build.yaml)):

```bash
pre-commit run --show-diff-on-failure --color always --all-files
make test lint build
```

## Pre-commit (required)

Contributions **must** pass [pre-commit](https://pre-commit.com). Install hooks once:

```bash
make pre-commit-install
```

Without Homebrew, install [`pre-commit`](https://pre-commit.com/#install) yourself first if `make pre-commit-install` cannot put it on your `PATH`.

**`brew bundle install`** puts **`pre-commit`**, **`gitleaks`**, and **`golangci-lint`** on your **`PATH`**. **`gitleaks`** is required for the secrets hook. The golangci-lint hook manages its own binary; **`make lint`** still expects **`golangci-lint`** on your **`PATH`**.

The `no-commit-to-branch` hook blocks commits on **`main`**—use a feature branch.

| Hook | Stage | Description |
|------|-------|-------------|
| `trailing-whitespace` | pre-commit | Strips trailing whitespace |
| `end-of-file-fixer` | pre-commit | Ensures files end with a newline |
| `check-yaml` | pre-commit | Validates YAML syntax |
| `no-commit-to-branch` | pre-commit | Prevents direct commits to `main` |
| `conventional-pre-commit` | commit-msg | [Conventional Commits](https://www.conventionalcommits.org/) (`feat`, `fix`, `docs`, …) |
| `golangci-lint` | pre-commit | Go lint ([.golangci.yaml](.golangci.yaml)) |
| `gitleaks` | pre-commit | Secret scan |

If a hook fails, fix or auto-fix, then commit again.

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
