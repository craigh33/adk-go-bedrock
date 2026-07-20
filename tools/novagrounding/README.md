# novagrounding

`novagrounding` enables Amazon Nova Web Grounding for Bedrock Converse.

Enable real-time web search for supported Nova models by adding [`novagrounding.Tool()`](tool.go) to `GenerateContentConfig.Tools`. This tool is Bedrock-specific. Use an applicable Bedrock region and an inference profile that supports Web Grounding (for example `us.amazon.nova-2-lite-v1:0`; see AWS docs for current model IDs). For current regional availability, check [Amazon Bedrock pricing](https://aws.amazon.com/bedrock/pricing/).

Converse request payloads use SystemTool name `nova_grounding`, while IAM policies for `bedrock:InvokeTool` may reference the resource identifier `amazon.nova_grounding`.

```go
import (
    "context"

    "google.golang.org/adk/v2/model"
    "google.golang.org/genai"

    "github.com/craigh33/adk-go-bedrock/tools/novagrounding"
)

func groundedAsk(ctx context.Context, llm model.LLM, question string) (*model.LLMResponse, error) {
    req := &model.LLMRequest{
        Contents: []*genai.Content{
            genai.NewContentFromText(question, genai.RoleUser),
        },
        Config: &genai.GenerateContentConfig{
            Tools:           []*genai.Tool{novagrounding.Tool()},
            MaxOutputTokens: 1024,
        },
    }
    var last *model.LLMResponse
    for resp, e := range llm.GenerateContent(ctx, req, false) {
        if e != nil {
            return nil, e
        }
        last = resp
    }
    return last, nil
}
```

Grounded replies include citation payloads under `genai.Part.PartMetadata` with key `"bedrock_citations"` (each entry may include `location.url`, `location.domain`, etc.). Retain and surface those citations in user-facing output per AWS guidance.

A runnable CLI lives at [`../../examples/bedrock-nova-grounding`](../../examples/bedrock-nova-grounding).
