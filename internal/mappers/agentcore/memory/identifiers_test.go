package memory

import (
	"strings"
	"testing"

	"google.golang.org/adk/v2/session"
)

func TestAgentCoreIDLimits(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name      string
		resolve   func(string) string
		maxLength int
	}{
		{"actor", func(id string) string { return ResolveActor("{userId}", "app", id) }, maxActorIDLength},
		{"session", ResolveSessionID, maxSessionIDLength},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			atLimit := strings.Repeat("a", tc.maxLength)
			if got := tc.resolve(atLimit); got != atLimit {
				t.Errorf("ID at length limit changed to %q", got)
			}
			for _, id := range []string{"", "_leading", "user@example.com", atLimit + "a"} {
				mapped := tc.resolve(id)
				if !strings.HasPrefix(mapped, encodedIDPrefix) || len(mapped) > tc.maxLength {
					t.Errorf("mapped ID %q is out of bounds", mapped)
				}
				if again := tc.resolve(id); again != mapped {
					t.Errorf("repeated mapping = %q, want %q", again, mapped)
				}
				if literal := tc.resolve(mapped); literal == mapped {
					t.Errorf("raw ID %q collides with the mapped ID for %q", literal, id)
				}
			}
		})
	}
}

func TestEventMetadataEncodesUnsupportedValues(t *testing.T) {
	t.Parallel()

	sess := testSession{appName: "my app", userID: "a?b", id: strings.Repeat("x", maxMetadataValueLength+1)}
	md := EventMetadata(sess, &session.Event{})
	assertMetadata(t, md, "app_name", "my app")
	if _, ok := md["app_name_encoding"]; ok {
		t.Error("raw metadata should have no encoding field")
	}
	assertMetadata(t, md, "user_id", "YT9i")
	assertMetadata(t, md, "user_id_encoding", "base64")
	assertMetadata(t, md, "session_id", "15eb95a462ee20bd91a415ae2d4aed341288186ddaa2b37908f7d592f0c3f85f")
	assertMetadata(t, md, "session_id_encoding", "sha256")
}
