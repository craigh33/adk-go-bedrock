.PHONY: generate check-generated

generate:
	go run github.com/oapi-codegen/oapi-codegen/v2/cmd/oapi-codegen@v2.8.1 --config oapi-codegen.yaml openapi.json

check-generated: generate
	git diff --exit-code -- types.gen.go
