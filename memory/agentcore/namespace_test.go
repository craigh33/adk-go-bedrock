package agentcorememory

import (
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/session"
	"google.golang.org/genai"
)

func TestMemoryActorUsesOneNamespaceSegment(t *testing.T) {
	t.Parallel()
	api := &fakeAPI{}
	svc, err := NewWithAPI(api, "mem-1", WithNamespacePath("/actors/{actorId}"))
	if err != nil {
		t.Fatal(err)
	}
	const userID = "group/member"
	sess := newSession(t, session.CreateRequest{AppName: "app", UserID: userID}, []seedEvent{
		{author: "user", role: genai.RoleUser, text: "hello"},
	})
	if err := svc.AddSessionToMemory(t.Context(), sess); err != nil {
		t.Fatal(err)
	}
	actorID := aws.ToString(api.createInputs[0].ActorId)
	if strings.Contains(actorID, "/") {
		t.Errorf("CreateEvent actor ID %q spans multiple namespace segments", actorID)
	}

	if _, err := svc.SearchMemory(t.Context(), &memory.SearchRequest{UserID: userID, Query: "q"}); err != nil {
		t.Fatal(err)
	}
	childPath := aws.ToString(api.retrieveIn.NamespacePath)
	if want := "/actors/" + actorID + "/"; childPath != want {
		t.Errorf("NamespacePath = %q, want %q", childPath, want)
	}
	if _, err := svc.SearchMemory(t.Context(), &memory.SearchRequest{UserID: "group", Query: "q"}); err != nil {
		t.Fatal(err)
	}
	groupPath := aws.ToString(api.retrieveIn.NamespacePath)
	if groupPath != "/actors/group/" {
		t.Errorf("NamespacePath = %q, want /actors/group/", groupPath)
	}
	if strings.HasPrefix(childPath, groupPath) {
		t.Errorf("group namespace %q also covers the other actor's namespace %q", groupPath, childPath)
	}
}

func TestSearchMemoryRejectsUnsafeNamespaceSubstitutions(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name string
		opt  Option
		req  memory.SearchRequest
	}{
		{
			name: "user wildcard",
			opt:  WithNamespacePath("/actors/{userId}/"),
			req:  memory.SearchRequest{UserID: "*"},
		},
		{
			name: "app path separator",
			opt:  WithNamespace("/apps/{appName}/actors/{actorId}/"),
			req:  memory.SearchRequest{AppName: "group/app", UserID: "user"},
		},
		{
			name: "empty user",
			opt:  WithNamespacePath("/actors/{userId}/"),
		},
		{
			name: "wildcard shares actor segment",
			opt:  WithNamespacePath("/actors/{actorId}*"),
			req:  memory.SearchRequest{UserID: "user"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			api := &fakeAPI{}
			svc, err := NewWithAPI(api, "mem-1", tc.opt)
			if err != nil {
				t.Fatal(err)
			}
			req := tc.req
			req.Query = "q"
			if _, err := svc.SearchMemory(t.Context(), &req); err == nil {
				t.Error("SearchMemory accepted an unsafe namespace value")
			}
			if api.retrieveIn != nil {
				t.Error("SearchMemory sent an unsafe namespace to AgentCore")
			}
		})
	}
}
