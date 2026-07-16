package mappers

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"net/netip"
	"path"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockagentcore"
	"github.com/aws/aws-sdk-go-v2/service/bedrockagentcore/types"
	"github.com/google/uuid"
)

const (
	agentCoreBrowserSessionName       = "adk-browser"
	agentCoreBrowserMinClientTokenLen = 33
	agentCoreBrowserMaxClientTokenLen = 256
	agentCoreBrowserMessageOverhead   = int64(1 << 20)
	base64DecodedBlockBytes           = int64(3)
	base64EncodedBlockBytes           = int64(4)
	agentCoreBrowserScreenshotPNG     = "png"
	agentCoreBrowserScreenshotJPEG    = "jpeg"
)

// AgentCoreBrowserStartParams is the tool-neutral input for starting a browser session.
type AgentCoreBrowserStartParams struct {
	BrowserIdentifier     string
	FunctionCallID        string
	SessionTimeoutSeconds int32
	ViewportWidth         int32
	ViewportHeight        int32
}

// AgentCoreBrowserStartInput maps browser configuration into the AgentCore service input.
func AgentCoreBrowserStartInput(p AgentCoreBrowserStartParams) *bedrockagentcore.StartBrowserSessionInput {
	in := &bedrockagentcore.StartBrowserSessionInput{
		BrowserIdentifier:     aws.String(p.BrowserIdentifier),
		ClientToken:           aws.String(AgentCoreBrowserClientToken(p.FunctionCallID)),
		Name:                  aws.String(agentCoreBrowserSessionName),
		SessionTimeoutSeconds: aws.Int32(p.SessionTimeoutSeconds),
	}
	if p.ViewportWidth > 0 {
		in.ViewPort = &types.ViewPort{
			Width:  aws.Int32(p.ViewportWidth),
			Height: aws.Int32(p.ViewportHeight),
		}
	}
	return in
}

// AgentCoreBrowserGetInput maps a browser identifier and session ID into the AgentCore service input.
func AgentCoreBrowserGetInput(browserIdentifier, sessionID string) *bedrockagentcore.GetBrowserSessionInput {
	return &bedrockagentcore.GetBrowserSessionInput{
		BrowserIdentifier: aws.String(browserIdentifier),
		SessionId:         aws.String(sessionID),
	}
}

// AgentCoreBrowserStopInput maps a browser session and ADK function-call ID into the AgentCore service input.
func AgentCoreBrowserStopInput(
	browserIdentifier string,
	sessionID string,
	functionCallID string,
) *bedrockagentcore.StopBrowserSessionInput {
	return &bedrockagentcore.StopBrowserSessionInput{
		BrowserIdentifier: aws.String(browserIdentifier),
		SessionId:         aws.String(sessionID),
		ClientToken:       aws.String(AgentCoreBrowserClientToken(functionCallID)),
	}
}

// AgentCoreBrowserAutomationEndpoint returns the private automation stream endpoint.
func AgentCoreBrowserAutomationEndpoint(streams *types.BrowserSessionStream) string {
	if streams == nil || streams.AutomationStream == nil {
		return ""
	}
	return aws.ToString(streams.AutomationStream.StreamEndpoint)
}

// AgentCoreBrowserLiveViewEndpoint returns the model-safe live-view stream endpoint.
func AgentCoreBrowserLiveViewEndpoint(streams *types.BrowserSessionStream) string {
	if streams == nil || streams.LiveViewStream == nil {
		return ""
	}
	return aws.ToString(streams.LiveViewStream.StreamEndpoint)
}

// AgentCoreBrowserTimeValue maps an optional service timestamp to an RFC3339 string.
func AgentCoreBrowserTimeValue(value *time.Time) string {
	if value == nil {
		return ""
	}
	return value.Format(time.RFC3339Nano)
}

// AgentCoreBrowserInt32Value maps an optional service integer to its zero value when absent.
func AgentCoreBrowserInt32Value(value *int32) int32 {
	if value == nil {
		return 0
	}
	return *value
}

// AgentCoreBrowserClientToken maps an ADK function-call ID to AgentCore's clientToken pattern.
func AgentCoreBrowserClientToken(functionCallID string) string {
	base := agentCoreBrowserSanitizeToken(functionCallID)
	if base == "" {
		return uuid.NewString()
	}
	if len(base) >= agentCoreBrowserMinClientTokenLen && len(base) <= agentCoreBrowserMaxClientTokenLen {
		return base
	}
	sum := sha256.Sum256([]byte(functionCallID))
	hash := hex.EncodeToString(sum[:])
	maxBase := agentCoreBrowserMaxClientTokenLen - len(hash) - 1
	if len(base) > maxBase {
		base = strings.TrimRight(base[:maxBase], "-")
	}
	return base + "-" + hash
}

func agentCoreBrowserSanitizeToken(value string) string {
	var result strings.Builder
	lastHyphen := false
	for _, r := range strings.TrimSpace(value) {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') {
			result.WriteRune(r)
			lastHyphen = false
			continue
		}
		if result.Len() > 0 && !lastHyphen {
			result.WriteByte('-')
			lastHyphen = true
		}
	}
	return strings.TrimRight(result.String(), "-")
}

// AgentCoreBrowserScreenshotFormat maps a requested format or filename extension to CDP and MIME values.
func AgentCoreBrowserScreenshotFormat(format, fileName string) (string, string, error) {
	if format == "" {
		fileName = strings.ToLower(fileName)
		switch {
		case strings.HasSuffix(fileName, ".jpg"), strings.HasSuffix(fileName, ".jpeg"):
			format = agentCoreBrowserScreenshotJPEG
		case strings.HasSuffix(fileName, ".png"):
			format = agentCoreBrowserScreenshotPNG
		}
	}
	switch strings.ToLower(format) {
	case "", agentCoreBrowserScreenshotPNG:
		return agentCoreBrowserScreenshotPNG, mimeImagePNG, nil
	case agentCoreBrowserScreenshotJPEG, "jpg":
		return agentCoreBrowserScreenshotJPEG, mimeImageJPEG, nil
	default:
		return "", "", fmt.Errorf("format must be png, jpeg, or jpg, got %q", format)
	}
}

// AgentCoreBrowserValidateScreenshotFileName validates that an artifact extension matches its format.
func AgentCoreBrowserValidateScreenshotFileName(fileName, format string) error {
	extension := strings.ToLower(path.Ext(fileName))
	switch extension {
	case "":
		return nil
	case ".png":
		if format == agentCoreBrowserScreenshotPNG {
			return nil
		}
	case ".jpg", ".jpeg":
		if format == agentCoreBrowserScreenshotJPEG {
			return nil
		}
	default:
		return fmt.Errorf("file_name has unsupported extension %q", extension)
	}
	return errors.New("file_name extension does not match format")
}

// AgentCoreBrowserNormalizeHosts normalizes host policy entries for suffix matching.
func AgentCoreBrowserNormalizeHosts(name string, hosts []string) ([]string, error) {
	result := make([]string, 0, len(hosts))
	for _, raw := range hosts {
		host := strings.TrimPrefix(strings.TrimSpace(raw), "*.")
		host = strings.TrimPrefix(host, ".")
		host = AgentCoreBrowserNormalizeHost(host)
		if host == "" {
			continue
		}
		if strings.ContainsAny(host, "*/?#@ \t\r\n") || strings.Contains(host, "..") {
			return nil, fmt.Errorf("agentcorebrowser: %s contains invalid host %q", name, raw)
		}
		if strings.Contains(host, ":") {
			host = strings.Trim(host, "[]")
			hostWithoutZone, _, _ := strings.Cut(host, "%")
			if _, err := netip.ParseAddr(hostWithoutZone); err != nil {
				return nil, fmt.Errorf("agentcorebrowser: %s contains invalid host %q", name, raw)
			}
		}
		result = append(result, host)
	}
	return result, nil
}

// AgentCoreBrowserNormalizeHost normalizes a browser hostname for policy comparison.
func AgentCoreBrowserNormalizeHost(host string) string {
	return strings.TrimSuffix(strings.ToLower(strings.TrimSpace(host)), ".")
}

// AgentCoreBrowserHostMatches reports whether a hostname equals or is a subdomain of a policy entry.
func AgentCoreBrowserHostMatches(patterns []string, host string) bool {
	for _, pattern := range patterns {
		if host == pattern || strings.HasSuffix(host, "."+pattern) {
			return true
		}
	}
	return false
}

// AgentCoreBrowserRequiresExplicitAllow reports whether a host is non-public or locally addressed.
func AgentCoreBrowserRequiresExplicitAllow(host string) bool {
	if host == "localhost" || strings.HasSuffix(host, ".localhost") {
		return true
	}
	host, _, _ = strings.Cut(host, "%")
	addr, err := netip.ParseAddr(host)
	if err != nil {
		return agentCoreBrowserLooksLikeLegacyIPv4(host)
	}
	addr = addr.Unmap()
	if !addr.IsGlobalUnicast() {
		return true
	}
	nonPublicIPPrefixes := [...]netip.Prefix{
		netip.MustParsePrefix("0.0.0.0/8"),
		netip.MustParsePrefix("10.0.0.0/8"),
		netip.MustParsePrefix("100.64.0.0/10"),
		netip.MustParsePrefix("127.0.0.0/8"),
		netip.MustParsePrefix("169.254.0.0/16"),
		netip.MustParsePrefix("172.16.0.0/12"),
		netip.MustParsePrefix("192.0.0.0/24"),
		netip.MustParsePrefix("192.0.2.0/24"),
		netip.MustParsePrefix("192.168.0.0/16"),
		netip.MustParsePrefix("198.18.0.0/15"),
		netip.MustParsePrefix("198.51.100.0/24"),
		netip.MustParsePrefix("203.0.113.0/24"),
		netip.MustParsePrefix("224.0.0.0/4"),
		netip.MustParsePrefix("240.0.0.0/4"),
		netip.MustParsePrefix("::/128"),
		netip.MustParsePrefix("::1/128"),
		netip.MustParsePrefix("64:ff9b:1::/48"),
		netip.MustParsePrefix("100::/64"),
		netip.MustParsePrefix("2001:db8::/32"),
		netip.MustParsePrefix("fc00::/7"),
		netip.MustParsePrefix("fe80::/10"),
		netip.MustParsePrefix("ff00::/8"),
	}
	for _, prefix := range nonPublicIPPrefixes {
		if prefix.Contains(addr) {
			return true
		}
	}
	return false
}

func agentCoreBrowserLooksLikeLegacyIPv4(host string) bool {
	for part := range strings.SplitSeq(host, ".") {
		if part == "" {
			return false
		}
		digits := part
		base := 10
		if len(part) > 2 && strings.EqualFold(part[:2], "0x") {
			digits = part[2:]
			base = 16
		}
		if _, err := strconv.ParseUint(digits, base, 32); err != nil {
			return false
		}
	}
	return true
}

// AgentCoreBrowserAutomationReadLimit derives a bounded WebSocket message size from configured output limits.
func AgentCoreBrowserAutomationReadLimit(maxScreenshot int64, maxText int) (int64, error) {
	const maxInt64 = int64(^uint64(0) >> 1)
	groups := maxScreenshot / base64DecodedBlockBytes
	if groups > maxInt64/base64EncodedBlockBytes {
		return 0, errors.New("agentcorebrowser: MaxScreenshotBytes is too large")
	}
	base64Bytes := groups * base64EncodedBlockBytes
	if maxScreenshot%base64DecodedBlockBytes != 0 {
		if base64Bytes > maxInt64-base64EncodedBlockBytes {
			return 0, errors.New("agentcorebrowser: MaxScreenshotBytes is too large")
		}
		base64Bytes += base64EncodedBlockBytes
	}
	textBytes := int64(maxText)
	if textBytes > maxInt64/6 {
		return 0, errors.New("agentcorebrowser: MaxTextBytes is too large")
	}
	textBytes *= 6
	limit := max(base64Bytes, textBytes)
	if limit > maxInt64-agentCoreBrowserMessageOverhead {
		return 0, errors.New("agentcorebrowser: configured response limits are too large")
	}
	return limit + agentCoreBrowserMessageOverhead, nil
}

// AgentCoreBrowserTruncateUTF8 caps text by bytes without returning invalid UTF-8.
func AgentCoreBrowserTruncateUTF8(value string, maxBytes int) (string, bool) {
	if maxBytes <= 0 || len(value) <= maxBytes {
		return value, false
	}
	result := []byte(value[:maxBytes])
	for !utf8.Valid(result) && len(result) > 0 {
		result = result[:len(result)-1]
	}
	return string(result), true
}
