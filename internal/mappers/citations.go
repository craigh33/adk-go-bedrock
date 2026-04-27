package mappers

import (
	"fmt"
	"strings"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"google.golang.org/genai"
)

// PartMetadataKeyBedrockCitations is set on genai.Part.PartMetadata for citation segments.
const PartMetadataKeyBedrockCitations = "bedrock_citations"

func citationsContentBlockToPart(b *types.CitationsContentBlock) (*genai.Part, error) {
	if b == nil {
		return nil, errSkipPart
	}
	var text strings.Builder
	for _, c := range b.Content {
		if c == nil {
			continue
		}
		if v, ok := c.(*types.CitationGeneratedContentMemberText); ok {
			text.WriteString(v.Value)
		}
	}
	cites := make([]any, 0, len(b.Citations))
	for i := range b.Citations {
		cites = append(cites, citationToMap(&b.Citations[i]))
	}
	if len(cites) == 0 {
		if text.String() != "" {
			return &genai.Part{Text: text.String()}, nil
		}
		return nil, errSkipPart
	}
	meta := map[string]any{
		PartMetadataKeyBedrockCitations: cites,
	}
	part := &genai.Part{PartMetadata: meta}
	if text.String() != "" {
		part.Text = text.String()
	}
	return part, nil
}

func citationToMap(c *types.Citation) map[string]any {
	if c == nil {
		return map[string]any{}
	}
	m := map[string]any{}
	if c.Title != nil {
		m["title"] = *c.Title
	}
	if c.Source != nil {
		m["source"] = *c.Source
	}
	if loc := citationLocationToMap(c.Location); len(loc) > 0 {
		m["location"] = loc
	}
	if len(c.SourceContent) > 0 {
		var texts []string
		for _, sc := range c.SourceContent {
			if sc == nil {
				continue
			}
			if v, ok := sc.(*types.CitationSourceContentMemberText); ok {
				texts = append(texts, v.Value)
			}
		}
		if len(texts) > 0 {
			m["sourceContent"] = texts
		}
	}
	return m
}

func citationLocationToMap(loc types.CitationLocation) map[string]any {
	if loc == nil {
		return map[string]any{}
	}
	switch v := loc.(type) {
	case *types.CitationLocationMemberWeb:
		out := map[string]any{"type": "web"}
		if v.Value.Url != nil {
			out["url"] = *v.Value.Url
		}
		if v.Value.Domain != nil {
			out["domain"] = *v.Value.Domain
		}
		return out
	case *types.CitationLocationMemberDocumentChar:
		return map[string]any{"type": "documentChar", "value": fmt.Sprintf("%+v", v.Value)}
	case *types.CitationLocationMemberDocumentChunk:
		return map[string]any{"type": "documentChunk", "value": fmt.Sprintf("%+v", v.Value)}
	case *types.CitationLocationMemberDocumentPage:
		return map[string]any{"type": "documentPage", "value": fmt.Sprintf("%+v", v.Value)}
	case *types.CitationLocationMemberSearchResultLocation:
		return map[string]any{"type": "searchResultLocation", "value": fmt.Sprintf("%+v", v.Value)}
	default:
		return map[string]any{"type": fmt.Sprintf("%T", loc)}
	}
}

// CitationsDeltaToMap converts a streaming citation delta to a JSON-friendly map.
func CitationsDeltaToMap(d types.CitationsDelta) map[string]any {
	m := map[string]any{}
	if d.Title != nil {
		m["title"] = *d.Title
	}
	if d.Source != nil {
		m["source"] = *d.Source
	}
	if loc := citationLocationToMap(d.Location); len(loc) > 0 {
		m["location"] = loc
	}
	var srcText strings.Builder
	for _, sc := range d.SourceContent {
		if sc.Text != nil {
			srcText.WriteString(*sc.Text)
		}
	}
	if srcText.Len() > 0 {
		m["sourceContentText"] = srcText.String()
	}
	return m
}
