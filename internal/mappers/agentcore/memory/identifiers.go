package memory

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"regexp"
	"strings"

	"github.com/aws/aws-sdk-go-v2/service/bedrockagentcore/types"
)

const (
	maxActorIDLength       = 255
	maxSessionIDLength     = 100
	maxMetadataValueLength = 256
	encodedIDPrefix        = "adk-id-"
)

// Actor IDs use a subset of the CreateEvent pattern so each actor occupies
// one segment in strategy namespaces.
var (
	actorIDPattern       = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_-]*(:[A-Za-z0-9_-]+)*$`)
	sessionIDPattern     = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_-]*$`)
	metadataValuePattern = regexp.MustCompile(`^[A-Za-z0-9\s\x0B._:/=+@-]*$`)
)

// ResolveActor expands the {userId} and {appName} placeholders and converts the
// result to an AgentCore actor ID.
func ResolveActor(template, appName, userID string) string {
	id := strings.NewReplacer(
		"{userId}", userID,
		"{appName}", appName,
	).Replace(template)
	return agentCoreID(id, actorIDPattern, maxActorIDLength)
}

// ResolveSessionID converts an ADK session ID to an AgentCore session ID.
func ResolveSessionID(id string) string {
	return agentCoreID(id, sessionIDPattern, maxSessionIDLength)
}

func agentCoreID(id string, pattern *regexp.Regexp, maxLength int) string {
	// Reserve the digest prefix so a literal ID cannot match another ID's digest.
	if len(id) <= maxLength && !strings.HasPrefix(id, encodedIDPrefix) && pattern.MatchString(id) {
		return id
	}
	return encodedIDPrefix + digest(id)
}

func addMetadata(md map[string]types.MetadataValue, key, value string) {
	switch {
	case len(value) <= maxMetadataValueLength && metadataValuePattern.MatchString(value):
	case base64.StdEncoding.EncodedLen(len(value)) <= maxMetadataValueLength:
		value = base64.StdEncoding.EncodeToString([]byte(value))
		md[key+"_encoding"] = &types.MetadataValueMemberStringValue{Value: "base64"}
	default:
		value = digest(value)
		md[key+"_encoding"] = &types.MetadataValueMemberStringValue{Value: "sha256"}
	}
	md[key] = &types.MetadataValueMemberStringValue{Value: value}
}

func digest(value string) string {
	sum := sha256.Sum256([]byte(value))
	return hex.EncodeToString(sum[:])
}
