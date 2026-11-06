.PHONY: pre-commit-install pre-commit test build lint generate-decider check-generated-decider

# Install pre-commit hooks (requires pre-commit to be installed).
pre-commit-install:
	$(MAKE) pre-commit

# Install pre-commit hooks (requires pre-commit to be installed).
pre-commit:
	@command -v pre-commit >/dev/null 2>&1 || brew install pre-commit
	pre-commit install --install-hooks -t pre-commit -t commit-msg


# Run unit tests for all packages.
test:
	go test ./... -count=1

# Compile all packages (no test run).
build:
	go build ./...

# Run golangci-lint (see .golangci.yaml).
lint:
	golangci-lint run ./...

generate-decider:
	uv run --locked api/decider/export.py
	go run github.com/oapi-codegen/oapi-codegen/v2/cmd/oapi-codegen@v2.8.1 --config api/decider/oapi-codegen.yaml api/decider/openapi.json

check-generated-decider: generate-decider
	git diff --exit-code -- api/decider/openapi.json internal/models/agentcore/decider/types.gen.go
