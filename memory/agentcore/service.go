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
	api MemoryAPI
	cfg serviceConfig
}

// AddSessionToMemory writes the session's events to AgentCore Memory. It emits one
// CreateEvent per text-bearing event, preserving each event's role and timestamp;
// events without textual content are skipped. Events with IDs use a stable
// CreateEvent client token so repeated calls on a growing session are
// idempotent.
//
// AddSessionToMemory implements [memory.Service].
func (s *Service) AddSessionToMemory(ctx context.Context, sess session.Session) error {
	if sess == nil {
		return errors.New("agentcorememory: session is nil")
	}
	actorID := memorymappers.ResolveActor(s.cfg.actorIDTemplate, sess.AppName(), sess.UserID())
	sessionID := memorymappers.ResolveSessionID(sess.ID())

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
			MemoryId:       aws.String(s.cfg.memoryID),
			ActorId:        aws.String(actorID),
			SessionId:      aws.String(sessionID),
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
			in.ClientToken = aws.String(memorymappers.EventClientToken(s.cfg.memoryID, actorID, sess, event.ID))
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
// [WithNamespace] or [WithNamespacePath] must be provided.
//
// SearchMemory implements [memory.Service].
func (s *Service) SearchMemory(ctx context.Context, req *memory.SearchRequest) (*memory.SearchResponse, error) {
	if req == nil {
		return nil, errors.New("agentcorememory: search request is nil")
	}
	template := s.cfg.namespace
	isNamespacePath := s.cfg.namespacePath != ""
	if isNamespacePath {
		template = s.cfg.namespacePath
	}
	if template == "" {
		return nil, errors.New("agentcorememory: Namespace or NamespacePath is required to search")
	}
	actorID := memorymappers.ResolveActor(s.cfg.actorIDTemplate, req.AppName, req.UserID)
	namespace, err := memorymappers.ResolveNamespace(template, req.AppName, req.UserID, actorID)
	if err != nil {
		return nil, fmt.Errorf("agentcorememory: resolve search namespace: %w", err)
	}

	in := &bedrockagentcore.RetrieveMemoryRecordsInput{
		MemoryId: aws.String(s.cfg.memoryID),
		SearchCriteria: &types.SearchCriteria{
			SearchQuery:     aws.String(req.Query),
			MetadataFilters: s.cfg.metadataFilters,
		},
	}
	if s.cfg.strategyID != "" {
		in.SearchCriteria.MemoryStrategyId = aws.String(s.cfg.strategyID)
	}
	if s.cfg.topK > 0 {
		in.SearchCriteria.TopK = aws.Int32(s.cfg.topK)
	}
	if s.cfg.maxResults > 0 {
		in.MaxResults = aws.Int32(s.cfg.maxResults)
	}

	if isNamespacePath {
		in.NamespacePath = aws.String(namespace)
	} else {
		in.Namespace = aws.String(namespace)
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
