// Package agentcorememory implements the ADK [memory.Service] interface using Amazon
// Bedrock AgentCore Memory.
//
// It writes ADK session events to an AgentCore Memory resource with CreateEvent
// and searches long-term memory with RetrieveMemoryRecords, so agents running on
// Bedrock can persist and recall user-scoped memory across sessions through the
// same [runner.Config.MemoryService] hook used for other providers.
//
// Long-term memory extraction in AgentCore is asynchronous: events written with
// [Service.AddSessionToMemory] are not immediately returned by
// [Service.SearchMemory]. Allow time for extraction to complete before expecting
// freshly written events to be searchable.
//
// The AgentCore Memory resource (and its strategies/namespaces) must be
// provisioned out of band. Callers need IAM permission for
// bedrock-agentcore:CreateEvent and bedrock-agentcore:RetrieveMemoryRecords.
package agentcorememory

import (
	"context"
	"errors"
	"fmt"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockagentcore"
	"github.com/aws/aws-sdk-go-v2/service/bedrockagentcore/types"
	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/session"

	memorymappers "github.com/craigh33/adk-go-bedrock/internal/mappers/agentcore/memory"
)

var _ memory.Service = (*Service)(nil)

// Service implements [memory.Service] against an AgentCore Memory resource.
type Service struct {
	api API
	cfg Config
}

// AddSessionToMemory writes the session's events to AgentCore Memory. It emits one
// CreateEvent per text-bearing event, preserving each event's role and timestamp;
// events without textual content are skipped. The event id is used as the
// CreateEvent client token so repeated calls on a growing session are idempotent.
//
// AddSessionToMemory implements [memory.Service].
func (s *Service) AddSessionToMemory(ctx context.Context, sess session.Session) error {
	if sess == nil {
		return errors.New("agentcorememory: session is nil")
	}
	actorID := memorymappers.ResolveActor(s.cfg.ActorIDTemplate, sess.AppName(), sess.UserID())

	for event := range sess.Events().All() {
		content := event.Content
		if content == nil {
			content = event.LLMResponse.Content
		}
		text := memorymappers.TextFromContent(content)
		if text == "" {
			continue
		}
		in := &bedrockagentcore.CreateEventInput{
			MemoryId:       aws.String(s.cfg.MemoryID),
			ActorId:        aws.String(actorID),
			SessionId:      aws.String(sess.ID()),
			EventTimestamp: aws.Time(memorymappers.EventTimestamp(event, sess)),
			Payload: []types.PayloadType{
				&types.PayloadTypeMemberConversational{
					Value: types.Conversational{
						Content: &types.ContentMemberText{Value: text},
						Role:    memorymappers.ToAgentCoreRole(content),
					},
				},
			},
			Metadata: memorymappers.EventMetadata(sess, event),
		}
		if event.ID != "" {
			in.ClientToken = aws.String(event.ID)
		}
		if _, err := s.api.CreateEvent(ctx, in); err != nil {
			return fmt.Errorf("agentcorememory: CreateEvent: %w", err)
		}
	}
	return nil
}

// SearchMemory retrieves memory records relevant to req.Query and maps them to
// [memory.Entry] values. The configured namespace (with {actorId}, {userId} and
// {appName} substituted from the request) scopes the search; either
// Config.Namespace or Config.NamespacePath must be set.
//
// SearchMemory implements [memory.Service].
func (s *Service) SearchMemory(ctx context.Context, req *memory.SearchRequest) (*memory.SearchResponse, error) {
	if req == nil {
		return nil, errors.New("agentcorememory: search request is nil")
	}
	actorID := memorymappers.ResolveActor(s.cfg.ActorIDTemplate, req.AppName, req.UserID)

	in := &bedrockagentcore.RetrieveMemoryRecordsInput{
		MemoryId: aws.String(s.cfg.MemoryID),
		SearchCriteria: &types.SearchCriteria{
			SearchQuery:     aws.String(req.Query),
			MetadataFilters: s.cfg.MetadataFilters,
		},
	}
	if s.cfg.StrategyID != "" {
		in.SearchCriteria.MemoryStrategyId = aws.String(s.cfg.StrategyID)
	}
	if s.cfg.TopK > 0 {
		in.SearchCriteria.TopK = aws.Int32(s.cfg.TopK)
	}
	if s.cfg.MaxResults > 0 {
		in.MaxResults = aws.Int32(s.cfg.MaxResults)
	}

	switch {
	case s.cfg.NamespacePath != "":
		in.NamespacePath = aws.String(
			memorymappers.ResolveNamespace(s.cfg.NamespacePath, req.AppName, req.UserID, actorID),
		)
	case s.cfg.Namespace != "":
		in.Namespace = aws.String(memorymappers.ResolveNamespace(s.cfg.Namespace, req.AppName, req.UserID, actorID))
	default:
		return nil, errors.New("agentcorememory: Namespace or NamespacePath is required to search")
	}

	out, err := s.api.RetrieveMemoryRecords(ctx, in)
	if err != nil {
		return nil, fmt.Errorf("agentcorememory: RetrieveMemoryRecords: %w", err)
	}

	res := &memory.SearchResponse{Memories: make([]memory.Entry, 0, len(out.MemoryRecordSummaries))}
	for _, rec := range out.MemoryRecordSummaries {
		res.Memories = append(res.Memories, memorymappers.RecordToEntry(rec))
	}
	return res, nil
}
