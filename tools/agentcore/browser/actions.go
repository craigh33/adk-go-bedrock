package browser

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"net/netip"
	"net/url"
	"strings"
	"time"

	v4 "github.com/aws/aws-sdk-go-v2/aws/signer/v4"
	"github.com/aws/aws-sdk-go-v2/service/bedrockagentcore/types"
	"github.com/coder/websocket"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/genai"

	browsermappers "github.com/craigh33/adk-go-bedrock/internal/agentcore/mappers/browser"
)

var errChromeExtensionRequest = errors.New(`url: scheme "chrome-extension" is not allowed`)

func (t *browserTool) runNavigate(
	ctx agent.Context,
	m map[string]any,
	session *browserSession,
) (map[string]any, error) {
	rawURL, err := requiredString(m, paramURL)
	if err != nil {
		return nil, err
	}
	if err := t.checkURL(ctx, rawURL, URLStageNavigate); err != nil {
		return nil, err
	}
	waitUntil := t.waitUntil
	rawWaitUntil, err := optionalStringArg(m, paramWaitUntil)
	if err != nil {
		return nil, err
	}
	if rawWaitUntil != "" {
		waitUntil, err = normalizeWaitUntil(WaitUntil(rawWaitUntil))
		if err != nil {
			return nil, fmt.Errorf("%s: %w", paramWaitUntil, err)
		}
	}
	waitForSelector, err := optionalStringArg(m, paramWaitForSelector)
	if err != nil {
		return nil, err
	}
	navCtx, cancel := context.WithTimeout(ctx, t.navigationTimeout)
	defer cancel()

	current, err := t.currentSession(navCtx, session)
	if err != nil {
		return nil, err
	}
	var streams *types.BrowserSessionStream
	if current == nil {
		started, err := t.startSession(navCtx, ctx, session)
		if err != nil {
			return nil, err
		}
		streams = started.Streams
	} else {
		streams = current.Streams
	}

	meta, err := t.navigateSessionPage(navCtx, streams, rawURL, waitUntil, waitForSelector)
	if err != nil {
		if current == nil {
			err = t.cleanupStartedSession(ctx, session, err)
		}
		return nil, err
	}
	return map[string]any{
		resultKeyStatus:    statusSuccess,
		paramAction:        actionNavigate,
		resultKeyBrowserID: t.browserIdentifier,
		paramURL:           meta.URL,
		resultKeyTitle:     meta.Title,
	}, nil
}

func (t *browserTool) navigateSessionPage(
	ctx context.Context,
	streams *types.BrowserSessionStream,
	rawURL string,
	waitUntil WaitUntil,
	waitForSelector string,
) (*pageMetadata, error) {
	cdp, err := t.openCDP(ctx, browsermappers.AgentCoreBrowserAutomationEndpoint(streams))
	if err != nil {
		return nil, err
	}
	defer cdp.close()
	return t.navigatePage(ctx, cdp, rawURL, waitUntil, waitForSelector)
}

func (t *browserTool) navigatePage(
	ctx context.Context,
	cdp *cdpConn,
	rawURL string,
	waitUntil WaitUntil,
	waitForSelector string,
) (*pageMetadata, error) {
	if err := cdp.navigate(ctx, rawURL, waitUntil, t.requestHandler, t.authHandler); err != nil {
		return nil, err
	}
	if waitForSelector != "" {
		if err := cdp.waitForSelector(ctx, waitForSelector); err != nil {
			return nil, err
		}
	}
	meta, err := cdp.pageMetadata(ctx)
	if err != nil {
		return nil, err
	}
	if err := t.checkURL(ctx, meta.URL, URLStageFinal); err != nil {
		return nil, fmt.Errorf("final url: %w", err)
	}
	return meta, nil
}

func (t *browserTool) runExtractText(
	ctx agent.Context,
	m map[string]any,
	session *browserSession,
) (map[string]any, error) {
	selector, err := optionalStringArg(m, paramSelector)
	if err != nil {
		return nil, err
	}
	waitForSelector, err := optionalStringArg(m, paramWaitForSelector)
	if err != nil {
		return nil, err
	}
	if selector == "" {
		selector = waitForSelector
	}
	actionCtx, cancel := context.WithTimeout(ctx, t.navigationTimeout)
	defer cancel()
	streams, err := t.sessionStreams(actionCtx, session)
	if err != nil {
		return nil, err
	}
	cdp, err := t.openCDP(actionCtx, browsermappers.AgentCoreBrowserAutomationEndpoint(streams))
	if err != nil {
		return nil, err
	}
	defer cdp.close()
	if err := cdp.enableRequestInterception(actionCtx, t.requestHandler, t.authHandler); err != nil {
		return nil, err
	}
	if waitForSelector != "" {
		if err := cdp.waitForSelector(actionCtx, waitForSelector); err != nil {
			return nil, err
		}
	}

	result, err := t.extractPageText(actionCtx, cdp, selector)
	if err != nil {
		return nil, err
	}
	return map[string]any{
		resultKeyStatus:    statusSuccess,
		paramAction:        actionExtractText,
		resultKeyBrowserID: t.browserIdentifier,
		"text":             result.Text,
		paramURL:           result.URL,
		resultKeyTitle:     result.Title,
		"truncated":        result.Truncated,
	}, nil
}

func (t *browserTool) extractPageText(ctx context.Context, cdp *cdpConn, selector string) (*textResult, error) {
	result, err := cdp.extractText(ctx, selector, t.maxTextBytes)
	if err != nil {
		return nil, err
	}
	if err := t.checkURL(ctx, result.URL, URLStageCurrent); err != nil {
		return nil, fmt.Errorf("current url: %w", err)
	}
	text, truncated := browsermappers.AgentCoreBrowserTruncateUTF8(result.Text, t.maxTextBytes)
	result.Text = text
	result.Truncated = result.Truncated || truncated
	return result, nil
}

func (t *browserTool) runScreenshot(
	ctx agent.Context,
	m map[string]any,
	session *browserSession,
) (map[string]any, error) {
	fileName, err := optionalStringArg(m, paramFileName)
	if err != nil {
		return nil, err
	}
	requestedFormat, err := optionalStringArg(m, paramFormat)
	if err != nil {
		return nil, err
	}
	file, err := resolveScreenshotFile(fileName, requestedFormat)
	if err != nil {
		return nil, err
	}
	fullPage, err := optionalBool(m, paramFullPage, true)
	if err != nil {
		return nil, err
	}
	quality, err := optionalInt(m, paramQuality)
	if err != nil {
		return nil, err
	}
	if quality != nil {
		if *quality < 0 || *quality > 100 {
			return nil, errors.New("quality must be between 0 and 100")
		}
		if file.format == screenshotFormatPNG {
			return nil, errors.New("quality is only supported for JPEG screenshots")
		}
	}
	artifacts := ctx.Artifacts()
	if artifacts == nil {
		return nil, errors.New("artifact service is unavailable")
	}
	actionCtx, cancel := context.WithTimeout(ctx, t.navigationTimeout)
	defer cancel()
	streams, err := t.sessionStreams(actionCtx, session)
	if err != nil {
		return nil, err
	}
	cdp, err := t.openCDP(actionCtx, browsermappers.AgentCoreBrowserAutomationEndpoint(streams))
	if err != nil {
		return nil, err
	}
	defer cdp.close()
	if err := cdp.enableRequestInterception(actionCtx, t.requestHandler, t.authHandler); err != nil {
		return nil, err
	}

	meta, version, err := t.savePageScreenshot(actionCtx, cdp, artifacts, file, fullPage, quality)
	if err != nil {
		return nil, err
	}
	return map[string]any{
		resultKeyStatus:    statusSuccess,
		paramAction:        actionScreenshot,
		resultKeyBrowserID: t.browserIdentifier,
		paramFileName:      file.name,
		"version":          version,
		"mime_type":        file.mimeType,
		paramURL:           meta.URL,
		resultKeyTitle:     meta.Title,
	}, nil
}

type screenshotFile struct {
	name     string
	format   string
	mimeType string
}

func resolveScreenshotFile(fileName, requestedFormat string) (screenshotFile, error) {
	fileName = strings.TrimSpace(fileName)
	requestedFormat = strings.TrimSpace(requestedFormat)
	if strings.ContainsAny(fileName, `/\`) {
		return screenshotFile{}, errors.New("file_name cannot contain path separators")
	}
	format, mimeType, err := browsermappers.AgentCoreBrowserScreenshotFormat(requestedFormat, fileName)
	if err != nil {
		return screenshotFile{}, err
	}
	if fileName == "" {
		fileName = "browser_screenshot." + format
	}
	if err := browsermappers.AgentCoreBrowserValidateScreenshotFileName(fileName, format); err != nil {
		return screenshotFile{}, err
	}
	return screenshotFile{name: fileName, format: format, mimeType: mimeType}, nil
}

func (t *browserTool) savePageScreenshot(
	ctx context.Context,
	cdp *cdpConn,
	artifacts agent.Artifacts,
	file screenshotFile,
	fullPage bool,
	quality *int,
) (*pageMetadata, int64, error) {
	meta, err := cdp.pageMetadata(ctx)
	if err != nil {
		return nil, 0, err
	}
	if err := t.checkURL(ctx, meta.URL, URLStageCurrent); err != nil {
		return nil, 0, fmt.Errorf("current url: %w", err)
	}
	data, err := cdp.screenshot(ctx, file.format, fullPage, quality, t.maxScreenshotBytes)
	if err != nil {
		return nil, 0, err
	}
	meta, err = cdp.pageMetadata(ctx)
	if err != nil {
		return nil, 0, err
	}
	if err := t.checkURL(ctx, meta.URL, URLStageCurrent); err != nil {
		return nil, 0, fmt.Errorf("current url after capture: %w", err)
	}
	if int64(len(data)) > t.maxScreenshotBytes {
		return nil, 0, fmt.Errorf("screenshot exceeds MaxScreenshotBytes (%d)", t.maxScreenshotBytes)
	}
	saveResp, err := artifacts.Save(ctx, file.name, genai.NewPartFromBytes(data, file.mimeType))
	if err != nil {
		return nil, 0, fmt.Errorf("save artifact %q: %w", file.name, err)
	}
	return meta, saveResp.Version, nil
}

func (t *browserTool) openCDP(ctx context.Context, endpoint string) (*cdpConn, error) {
	if endpoint == "" {
		return nil, errors.New("browser session has no automation stream endpoint")
	}
	headers, err := t.signedWebSocketHeaders(ctx, endpoint)
	if err != nil {
		return nil, err
	}
	conn, resp, err := t.dialer.Dial(ctx, endpoint, &websocket.DialOptions{HTTPHeader: headers})
	if resp != nil && resp.Body != nil {
		_ = resp.Body.Close()
	}
	if err != nil {
		return nil, fmt.Errorf("connect automation stream: %w", err)
	}
	if conn == nil {
		return nil, errors.New("connect automation stream: dialer returned a nil connection")
	}
	conn.SetReadLimit(t.automationReadLimit)
	return &cdpConn{conn: conn}, nil
}

func (t *browserTool) signedWebSocketHeaders(ctx context.Context, endpoint string) (http.Header, error) {
	u, err := url.Parse(endpoint)
	if err != nil {
		return nil, fmt.Errorf("parse automation stream endpoint: %w", err)
	}
	signURL := *u
	switch signURL.Scheme {
	case "wss":
		signURL.Scheme = schemeHTTPS
	case "ws":
		signURL.Scheme = schemeHTTP
	default:
		return nil, fmt.Errorf("automation stream endpoint scheme must be ws or wss, got %q", signURL.Scheme)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, signURL.String(), nil)
	if err != nil {
		return nil, fmt.Errorf("build automation stream request: %w", err)
	}
	creds, err := t.credentials.Retrieve(ctx)
	if err != nil {
		return nil, fmt.Errorf("retrieve AWS credentials: %w", err)
	}
	if err := v4.NewSigner().
		SignHTTP(ctx, creds, req, emptyPayloadSHA256, serviceID, t.region, time.Now()); err != nil {
		return nil, fmt.Errorf("sign automation stream request: %w", err)
	}
	return req.Header, nil
}

type originalURLContextKey struct{}

func (t *browserTool) checkURL(ctx context.Context, raw string, stage URLStage) error {
	check, err := parseURLCheck(raw, stage)
	if err != nil {
		return err
	}
	ctx = context.WithValue(ctx, originalURLContextKey{}, check.URL.String())
	return t.urlHandler(ctx, check)
}

func parseURLCheck(raw string, stage URLStage) (URLCheck, error) {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil {
		return URLCheck{}, fmt.Errorf("url: %w", err)
	}
	check := URLCheck{URL: *u, Stage: stage}
	if err := validateURLStructure(check); err != nil {
		return URLCheck{}, err
	}
	return check, nil
}

func validateURLStructure(check URLCheck) error {
	u := &check.URL
	if u.User != nil {
		return errors.New("url: user info is not allowed")
	}
	switch check.Stage {
	case URLStageNavigate, URLStageCurrent, URLStageFinal:
		if u.Scheme != schemeHTTP && u.Scheme != schemeHTTPS {
			return fmt.Errorf("url: scheme must be http or https, got %q", u.Scheme)
		}
	case URLStageRequest:
		switch u.Scheme {
		case schemeHTTP, schemeHTTPS:
		case schemeData, schemeBlob:
			return nil
		case "chrome-extension":
			return errChromeExtensionRequest
		default:
			return fmt.Errorf("url: scheme %q is not allowed", u.Scheme)
		}
	default:
		return fmt.Errorf("url: invalid stage %q", check.Stage)
	}
	host := u.Hostname()
	if host == "" {
		return errors.New("url: host is required")
	}
	if strings.ContainsAny(host, `@/\`) {
		return fmt.Errorf("url: host %q contains invalid characters", host)
	}
	if strings.Contains(host, "%") {
		addr, err := netip.ParseAddr(host)
		if err != nil || addr.Zone() == "" {
			return fmt.Errorf("url: host %q contains an invalid percent escape", host)
		}
	}
	return nil
}

func (t *browserTool) handleURLCheck(ctx context.Context, check URLCheck) error {
	if err := validateURLStructure(check); err != nil {
		return err
	}
	if originalURL, ok := ctx.Value(originalURLContextKey{}).(string); ok && check.URL.String() != originalURL {
		return errors.New("URL middleware cannot rewrite URLs; use request middleware to rewrite browser requests")
	}
	u := &check.URL
	if check.Stage == URLStageRequest && (u.Scheme == schemeData || u.Scheme == schemeBlob) {
		return nil
	}
	host := browsermappers.AgentCoreBrowserNormalizeHost(u.Hostname())
	if browsermappers.AgentCoreBrowserHostMatches(t.deniedHosts, host) {
		return fmt.Errorf("url: host %q is denied", host)
	}
	if len(t.allowedHosts) > 0 && !browsermappers.AgentCoreBrowserHostMatches(t.allowedHosts, host) {
		return fmt.Errorf("url: host %q is not allowed", host)
	}
	if len(t.allowedHosts) == 0 && !t.allowUnrestrictedDNS {
		if _, err := netip.ParseAddr(host); err != nil {
			return fmt.Errorf("url: DNS host %q requires an AllowedHosts entry or AllowUnrestrictedDNS", host)
		}
	}
	if len(t.allowedHosts) == 0 && browsermappers.AgentCoreBrowserRequiresExplicitAllow(host) {
		return fmt.Errorf("url: host %q requires an explicit allowlist entry", host)
	}
	return nil
}

func (t *browserTool) handleBrowserRequest(
	ctx context.Context,
	request *Request,
) (*Response, error) {
	if err := t.checkURL(ctx, request.URL, URLStageRequest); err != nil {
		return nil, err
	}
	return nil, nil //nolint:nilnil // A nil response and error means continue unchanged.
}

func normalizeWaitUntil(value WaitUntil) (WaitUntil, error) {
	switch value {
	case "", WaitUntilLoad:
		return WaitUntilLoad, nil
	case WaitUntilDOMContentLoaded, WaitUntilNone:
		return value, nil
	default:
		return "", fmt.Errorf("WaitUntil must be load, dom_content_loaded, or none, got %q", value)
	}
}

func requiredString(m map[string]any, key string) (string, error) {
	value, err := optionalStringArg(m, key)
	if err != nil {
		return "", err
	}
	if value == "" {
		return "", fmt.Errorf("%s is required", key)
	}
	return value, nil
}

func optionalStringArg(m map[string]any, key string) (string, error) {
	value, ok := m[key]
	if !ok {
		return "", nil
	}
	result, ok := value.(string)
	if !ok {
		return "", fmt.Errorf("%s must be a string", key)
	}
	return strings.TrimSpace(result), nil
}

func optionalBool(m map[string]any, key string, defaultValue bool) (bool, error) {
	value, ok := m[key]
	if !ok {
		return defaultValue, nil
	}
	result, ok := value.(bool)
	if !ok {
		return false, fmt.Errorf("%s must be a boolean", key)
	}
	return result, nil
}

func optionalInt(m map[string]any, key string) (*int, error) {
	value, ok := m[key]
	if !ok {
		return nil, nil //nolint:nilnil // A nil pointer represents an omitted optional argument.
	}
	var result int64
	switch value := value.(type) {
	case int:
		result = int64(value)
	case int8:
		result = int64(value)
	case int16:
		result = int64(value)
	case int32:
		result = int64(value)
	case int64:
		result = value
	case uint:
		if uint64(value) > math.MaxInt64 {
			return nil, fmt.Errorf("%s must be an integer", key)
		}
		result = int64(value)
	case uint8:
		result = int64(value)
	case uint16:
		result = int64(value)
	case uint32:
		result = int64(value)
	case uint64:
		if value > math.MaxInt64 {
			return nil, fmt.Errorf("%s must be an integer", key)
		}
		result = int64(value)
	case float32:
		parsed, valid := integralInt64(float64(value))
		if !valid {
			return nil, fmt.Errorf("%s must be an integer", key)
		}
		result = parsed
	case float64:
		parsed, valid := integralInt64(value)
		if !valid {
			return nil, fmt.Errorf("%s must be an integer", key)
		}
		result = parsed
	case json.Number:
		parsed, err := value.Int64()
		if err != nil {
			return nil, fmt.Errorf("%s must be an integer", key)
		}
		result = parsed
	default:
		return nil, fmt.Errorf("%s must be an integer", key)
	}
	converted := int(result)
	if int64(converted) != result {
		return nil, fmt.Errorf("%s must be an integer", key)
	}
	return &converted, nil
}

func integralInt64(value float64) (int64, bool) {
	const int64UpperBound = float64(1 << 63)
	if math.IsNaN(value) || math.IsInf(value, 0) || math.Trunc(value) != value ||
		value < -int64UpperBound || value >= int64UpperBound {
		return 0, false
	}
	return int64(value), true
}
