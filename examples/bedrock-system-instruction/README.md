# bedrock-system-instruction example

This example demonstrates best practices for using system instructions with the Bedrock Converse provider, including role definition, output formatting, and behavioral control.

## Features

- System instructions for persona/role definition
- Output format control (JSON, markdown, structured text)
- Multi-turn conversations with consistent system context
- Conversation history with system instructions
- System instruction variations for different tasks

## Prerequisites

- `BEDROCK_MODEL_ID` set to a Bedrock model ID or inference profile ARN
- AWS credentials configured via the default chain
- AWS region configured (for example `AWS_REGION=us-east-1`)

## Run

```bash
make -C examples/bedrock-system-instruction run
```

## System Instruction Patterns

This example includes several patterns:

1. **Role Definition** - Define the AI's persona and expertise
2. **Output Formatting** - Specify JSON, markdown, or structured output
3. **Behavior Control** - Set constraints on tone, detail level, and approach
4. **Context Specification** - Provide domain-specific knowledge or constraints
