.PHONY: dev test build lint

# Run the bedrock-chat example (set BEDROCK_MODEL_ID and AWS auth as in README).
dev:
	go run ./examples/bedrock-chat

# Run the bedrock-web-ui example (set BEDROCK_MODEL_ID and AWS auth as in README).
dev-ui:
 	go run ./examples/bedrock-web-ui/main.go web api webui

# Run unit tests for all packages.
test:
	go test ./... -count=1

# Compile all packages (no test run).
build:
	go build ./...

# Run golangci-lint (see .golangci.yaml).
lint:
	golangci-lint run ./...
