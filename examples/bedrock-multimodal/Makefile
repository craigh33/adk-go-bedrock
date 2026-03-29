.PHONY: run

run:
	@if [ -z "$$BEDROCK_MODEL_ID" ]; then \
		echo "Error: BEDROCK_MODEL_ID is not set"; \
		echo "Usage: BEDROCK_MODEL_ID=<model-id> AWS_REGION=<region> make run"; \
		exit 1; \
	fi
	BEDROCK_MODEL_ID=$(BEDROCK_MODEL_ID) \
		AWS_REGION=$(AWS_REGION) \
		IMAGE_URL=$(IMAGE_URL) \
		go run .
