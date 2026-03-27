# bedrock-tool-calling example

This example demonstrates tool calling with the Bedrock Converse provider using function declarations. It executes a weather tool locally, then sends the tool result back to the model for a final answer.

## Features

- **Tool Definition**: Shows how to define function declarations in `genai.Tool`
- **Direct LLM Flow**: Uses two model calls (tool call + final response)
- **Tool Execution**: Demonstrates handling tool calls from the model
- **Response Processing**: Shows how to extract and process tool calls from LLM responses

## Prerequisites

- `BEDROCK_MODEL_ID` set to a Bedrock model ID or inference profile ARN (must support tool use)
- AWS credentials configured via the default chain
- AWS region configured (for example `AWS_REGION=us-east-1`)

## Run

```bash
make -C examples/bedrock-tool-calling run
```

Or pass a custom question:

```bash
make -C examples/bedrock-tool-calling run PROMPT='What is the weather in London?'
```

## How It Works

1. Define a `get_weather` tool that takes a city name parameter
2. Send an initial request with the weather function declaration
3. Detect and execute model function calls locally
4. Send tool results back as `FunctionResponse` parts
5. Print the model's final natural-language response

## Example Conversation

```
2026/03/27 10:07:29 BEDROCK_MODEL_ID is required (e.g. eu.amazon.nova-2-lite-v1:0) using default model
User: What's the weather like in Seattle?

Tool call: get_weather args=map[city:Seattle]
Tool result: Weather in Seattle: cloudy, 80°F

Assistant's final response:
The weather in Seattle is currently cloudy with a temperature of 80°F.

Finish reason: STOP
```
