.PHONY: run

run:
	@if [ -z "$$BEDROCK_MODEL_ID" ]; then \
		echo "Error: BEDROCK_MODEL_ID is not set"; \
		echo "Usage: BEDROCK_MODEL_ID=<model-id> AWS_REGION=<region> make run"; \
		exit 1; \
	fi
	go run ./examples/bedrock-system-instruction
