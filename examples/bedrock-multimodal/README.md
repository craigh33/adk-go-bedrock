# bedrock-multimodal example

This example demonstrates comprehensive multimodal capabilities using the Bedrock Converse provider, including image analysis, tool calling with rich media, extended reasoning, and conversation history management.

## Features

### Image Analysis
- **Basic Image Analysis**: Describe and analyze images with vision models
- **System Instructions**: Use detailed system prompts to guide image analysis
- **Extended Reasoning**: Ask for step-by-step visual analysis and reasoning
- **Image Constraints**: Handle Bedrock's image size, format, and placement constraints

### Conversation Management
- **Multi-turn Conversations**: Maintain conversation history with images
- **Image Reuse**: Send images in first turn, reference in follow-up turns
- **Context Preservation**: Keep system instructions across multiple turns

### Tool Calling
- **Image Tagging**: Demonstrate tool calling with images for structured output
- **Region Analysis**: Call functions that process specific image regions
- **Tool Response Media**: Include images in function responses

### Advanced Features
- **Multiple Images**: Compare and analyze multiple images in a single request
- **Streaming**: Stream image analysis responses for real-time feedback
- **Media in Tool Responses**: Include images and documents in tool responses
- **Image Constraints**: Work within Bedrock's media handling constraints

## Prerequisites

- `BEDROCK_MODEL_ID` set to a Bedrock model ID that supports vision (e.g., Claude 3.5 Sonnet, Nova Pro)
- AWS credentials configured via the default chain
- AWS region configured (for example `AWS_REGION=us-east-1`)

## Run

```bash
make -C examples/bedrock-multimodal run
```

Or with a custom image URL:

```bash
IMAGE_URL='https://example.com/image.jpg' make -C examples/bedrock-multimodal run
```

## Example Walkthrough

### 1. Basic Image Analysis
```go
req := &model.LLMRequest{
    Contents: []*genai.Content{
        {
            Role: genai.RoleUser,
            Parts: []*genai.Part{
                {Text: "Describe this image"},
                {InlineData: &genai.Blob{
                    MIMEType: "image/jpeg",
                    Data: imageData,
                }},
            },
        },
    },
}
```

### 2. Image Analysis with System Instructions
```go
Config: &genai.GenerateContentConfig{
    SystemInstruction: &genai.Content{
        Parts: []*genai.Part{
            {Text: "You are an expert visual analyst..."},
        },
    },
}
```

### 3. Tool Calling with Images
```go
Config: &genai.GenerateContentConfig{
    Tools: []*genai.Tool{imageTaggingTool},
}

// Listen for function calls
if part.FunctionCall != nil {
    // Process tool call
}
```

### 4. Multi-turn Conversations
```go
// First turn: send image
contents := []*genai.Content{
    {Role: genai.RoleUser, Parts: []*genai.Part{
        {Text: "Here's an image"},
        {InlineData: &genai.Blob{...}},
    }},
}

// Second turn: follow-up without re-sending image
contents = append(contents, &genai.Content{
    Role: genai.RoleUser,
    Parts: []*genai.Part{{Text: "What colors dominate?"}},
})
```

### 5. Streaming Image Analysis
```go
for resp := range llm.GenerateContent(ctx, req, true) { // true = streaming
    // Process streamed chunks
}
```

## Bedrock Multimodal Constraints

### Image Specifications
- **Formats**: JPEG, PNG, GIF, WebP
- **Placement**: Images only in user turns, not in model turns
- **Quantity**: Multiple images per request supported
- **Size**: Check AWS documentation for size limits per model

### Document Processing
- PDF and text documents can be processed where supported
- Documents follow similar placement constraints as images

### Audio & Video
- Some models support audio and video analysis
- Check model documentation for specific support

## Content Types Mapping

| ADK Type | Bedrock Support | Notes |
|----------|-----------------|-------|
| Image (InlineData) | ✅ | User turns only |
| Image (File URI) | ✅ | S3 URI format |
| Text | ✅ | Both user and model turns |
| Tool Use | ✅ | Model turn responses |
| Function Response | ✅ | User turn tool results |
| Document | ⚠️ | Model-dependent |
| Audio | ⚠️ | Model-dependent |
| Video | ⚠️ | Model-dependent |

## Error Handling

Common issues and solutions:

1. **Image Size**: Large images may exceed Bedrock limits. Try compressing.
2. **Format Issues**: Ensure images are JPEG, PNG, GIF, or WebP.
3. **Placement Errors**: Images only go in user turns; model turns cannot contain images.
4. **Model Compatibility**: Verify your model supports vision. Use `us.anthropic.claude-3-5-sonnet-20241022-v2:0` or similar.

## Token Usage

Image analysis consumes more tokens than text:
- Image token cost varies by model
- Multiple images increase token usage proportionally
- Check `UsageMetadata` in responses for exact counts

## Limitations

- Function declarations map to custom Bedrock tool specs
- Non-function tool variants (Google Search, Code Execution, Retrieval, MCP Servers, etc.) map to Bedrock system tools
- All supported tool variants are now sent to Bedrock; unsupported variants cause an error
- Rich media input follows Bedrock Converse constraints

## More Resources

- [Bedrock Converse API Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html)
- [Claude Vision Capabilities](https://docs.anthropic.com/vision/vision-intro)
- [ADK Model Documentation](https://pkg.go.dev/google.golang.org/adk/model)
