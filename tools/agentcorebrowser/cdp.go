package agentcorebrowser

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"net/http"
	"slices"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/gorilla/websocket"
)

type cdpConn struct {
	conn           *websocket.Conn
	nextID         int64
	pageSessionID  string
	events         []cdpMessage
	responses      map[int64]cdpMessage
	requestHandler RequestHandler
	authHandler    AuthHandler
}

type cdpMessage struct {
	ID        int64           `json:"id,omitempty"`
	SessionID string          `json:"sessionId,omitempty"`
	Method    string          `json:"method,omitempty"`
	Params    json.RawMessage `json:"params,omitempty"`
	Result    json.RawMessage `json:"result,omitempty"`
	Error     *cdpError       `json:"error,omitempty"`
}

type cdpError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

type runtimeEvaluateResult struct {
	Result struct {
		Value json.RawMessage `json:"value"`
	} `json:"result"`
	ExceptionDetails *runtimeExceptionDetails `json:"exceptionDetails,omitempty"`
}

type runtimeExceptionDetails struct {
	Text      string `json:"text,omitempty"`
	Exception *struct {
		Description string `json:"description,omitempty"`
	} `json:"exception,omitempty"`
}

type pageMetadata struct {
	URL   string `json:"url"`
	Title string `json:"title"`
}

type textResult struct {
	Text      string `json:"text"`
	URL       string `json:"url"`
	Title     string `json:"title"`
	Truncated bool   `json:"truncated"`
}

type pausedRequest struct {
	RequestID           string `json:"requestId"`
	ResourceType        string `json:"resourceType"`
	FrameID             string `json:"frameId"`
	NetworkID           string `json:"networkId"`
	RedirectedRequestID string `json:"redirectedRequestId"`
	Request             struct {
		URL      string            `json:"url"`
		Method   string            `json:"method"`
		Headers  map[string]string `json:"headers"`
		PostData *string           `json:"postData"`
	} `json:"request"`
}

type authRequired struct {
	RequestID    string `json:"requestId"`
	ResourceType string `json:"resourceType"`
	FrameID      string `json:"frameId"`
	Request      struct {
		URL      string            `json:"url"`
		Method   string            `json:"method"`
		Headers  map[string]string `json:"headers"`
		PostData *string           `json:"postData"`
	} `json:"request"`
	AuthChallenge struct {
		Source string `json:"source"`
		Origin string `json:"origin"`
		Scheme string `json:"scheme"`
		Realm  string `json:"realm"`
	} `json:"authChallenge"`
}

func (c *cdpConn) close() {
	_ = c.conn.Close()
}

func (c *cdpConn) navigate(
	ctx context.Context,
	rawURL string,
	waitUntil WaitUntil,
	handler RequestHandler,
	authHandler AuthHandler,
) error {
	sessionID, err := c.pageSession(ctx)
	if err != nil {
		return err
	}
	if err := c.enableRequestInterception(ctx, handler, authHandler); err != nil {
		return err
	}
	if _, err := c.call(ctx, "Page.enable", nil, sessionID); err != nil {
		return err
	}
	raw, err := c.call(ctx, "Page.navigate", map[string]any{paramURL: rawURL}, sessionID)
	if err != nil {
		return err
	}
	var result struct {
		ErrorText  string `json:"errorText"`
		IsDownload bool   `json:"isDownload"`
	}
	if err := json.Unmarshal(raw, &result); err != nil {
		return fmt.Errorf("parse navigation result: %w", err)
	}
	if result.ErrorText != "" {
		return fmt.Errorf("navigate: %s", result.ErrorText)
	}
	if result.IsDownload {
		return errors.New("navigate: URL started a download")
	}
	event := navigationEvent(waitUntil)
	if event == "" {
		return nil
	}
	return c.waitEvent(ctx, sessionID, event)
}

func navigationEvent(waitUntil WaitUntil) string {
	switch waitUntil {
	case WaitUntilNone:
		return ""
	case WaitUntilDOMContentLoaded:
		return "Page.domContentEventFired"
	case WaitUntilLoad:
		return "Page.loadEventFired"
	default:
		return ""
	}
}

func (c *cdpConn) enableRequestInterception(
	ctx context.Context,
	handler RequestHandler,
	authHandler AuthHandler,
) error {
	sessionID, err := c.pageSession(ctx)
	if err != nil {
		return err
	}
	c.requestHandler = handler
	c.authHandler = authHandler
	params := map[string]any{
		"patterns": []map[string]any{{
			"urlPattern":   "*",
			"requestStage": "Request",
		}},
	}
	if authHandler != nil {
		params["handleAuthRequests"] = true
	}
	_, err = c.call(ctx, "Fetch.enable", params, sessionID)
	if err != nil {
		c.requestHandler = nil
		c.authHandler = nil
	}
	return err
}

func (c *cdpConn) waitForSelector(ctx context.Context, selector string) error {
	sessionID, err := c.pageSession(ctx)
	if err != nil {
		return err
	}
	selectorJSON, _ := json.Marshal(selector)
	expr := `new Promise((resolve, reject) => {
const selector = ` + string(selectorJSON) + `;
let observer;
const find = () => {
    try {
        if (!document.querySelector(selector)) return false;
        if (observer) observer.disconnect();
        resolve(true);
        return true;
    } catch (error) {
        if (observer) observer.disconnect();
        reject(error);
        return true;
    }
};
if (find()) return;
observer = new MutationObserver(find);
observer.observe(document.documentElement || document, {attributes: true, childList: true, subtree: true});
})`
	raw, err := c.call(ctx, "Runtime.evaluate", map[string]any{
		cdpKeyExpression:    expr,
		"awaitPromise":      true,
		cdpKeyReturnByValue: true,
	}, sessionID)
	if err != nil {
		return err
	}
	var found bool
	if err := unmarshalRuntimeValue(raw, "wait_for_selector", &found); err != nil {
		return err
	}
	if !found {
		return fmt.Errorf("wait_for_selector %q did not resolve", selector)
	}
	return nil
}

func (c *cdpConn) extractText(ctx context.Context, selector string, maxBytes int) (*textResult, error) {
	sessionID, err := c.pageSession(ctx)
	if err != nil {
		return nil, err
	}
	selectorJSON := "null"
	if selector != "" {
		b, _ := json.Marshal(selector)
		selectorJSON = string(b)
	}
	expr := `(() => {
const selector = ` + selectorJSON + `;
const maxBytes = ` + strconv.Itoa(maxBytes) + `;
const node = selector ? document.querySelector(selector) : document.body;
let text = node ? (node.innerText || node.textContent || "") : "";
const bytes = new TextEncoder().encode(text);
const truncated = maxBytes > 0 && bytes.length > maxBytes;
if (truncated) {
    let end = maxBytes;
    while (end > 0 && (bytes[end] & 0xc0) === 0x80) {
        end--;
    }
    text = new TextDecoder().decode(bytes.subarray(0, end));
}
return {text, url: location.href, title: document.title, truncated};
})()`
	raw, err := c.call(ctx, "Runtime.evaluate", map[string]any{
		cdpKeyExpression:    expr,
		cdpKeyReturnByValue: true,
	}, sessionID)
	if err != nil {
		return nil, err
	}
	var result textResult
	if err := unmarshalRuntimeValue(raw, "extract_text", &result); err != nil {
		return nil, err
	}
	return &result, nil
}

func (c *cdpConn) pageMetadata(ctx context.Context) (*pageMetadata, error) {
	sessionID, err := c.pageSession(ctx)
	if err != nil {
		return nil, err
	}
	raw, err := c.call(ctx, "Runtime.evaluate", map[string]any{
		cdpKeyExpression:    `({url: location.href, title: document.title})`,
		cdpKeyReturnByValue: true,
	}, sessionID)
	if err != nil {
		return nil, err
	}
	var result pageMetadata
	if err := unmarshalRuntimeValue(raw, "page metadata", &result); err != nil {
		return nil, err
	}
	return &result, nil
}

func (c *cdpConn) screenshot(
	ctx context.Context,
	format string,
	fullPage bool,
	quality *int,
	maxBytes int64,
) ([]byte, error) {
	sessionID, err := c.pageSession(ctx)
	if err != nil {
		return nil, err
	}
	params := map[string]any{
		"format":                format,
		"captureBeyondViewport": fullPage,
	}
	if quality != nil {
		params[paramQuality] = *quality
	}
	raw, err := c.call(ctx, "Page.captureScreenshot", params, sessionID)
	if err != nil {
		return nil, err
	}
	var out struct {
		Data string `json:"data"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("parse screenshot result: %w", err)
	}
	decodedLen := base64DecodedLen(out.Data)
	if decodedLen > maxBytes {
		return nil, fmt.Errorf("screenshot exceeds MaxScreenshotBytes (%d)", maxBytes)
	}
	data, err := base64.StdEncoding.DecodeString(out.Data)
	if err != nil {
		return nil, fmt.Errorf("decode screenshot: %w", err)
	}
	if int64(len(data)) > maxBytes {
		return nil, fmt.Errorf("screenshot exceeds MaxScreenshotBytes (%d)", maxBytes)
	}
	return data, nil
}

func base64DecodedLen(encoded string) int64 {
	decodedLen := base64.StdEncoding.DecodedLen(len(encoded))
	if strings.HasSuffix(encoded, "=") {
		decodedLen--
	}
	if strings.HasSuffix(encoded, "==") {
		decodedLen--
	}
	return int64(max(decodedLen, 0))
}

func (c *cdpConn) pageSession(ctx context.Context) (string, error) {
	if c.pageSessionID != "" {
		return c.pageSessionID, nil
	}
	raw, err := c.call(ctx, "Target.getTargets", nil, "")
	if err != nil {
		return "", err
	}
	var targets struct {
		TargetInfos []struct {
			TargetID string `json:"targetId"`
			Type     string `json:"type"`
		} `json:"targetInfos"`
	}
	if err := json.Unmarshal(raw, &targets); err != nil {
		return "", fmt.Errorf("parse targets: %w", err)
	}
	var targetID string
	for _, info := range targets.TargetInfos {
		if info.Type == "page" {
			targetID = info.TargetID
			break
		}
	}
	if targetID == "" {
		raw, err = c.call(ctx, "Target.createTarget", map[string]any{paramURL: "about:blank"}, "")
		if err != nil {
			return "", err
		}
		var created struct {
			TargetID string `json:"targetId"`
		}
		if err := json.Unmarshal(raw, &created); err != nil {
			return "", fmt.Errorf("parse created target: %w", err)
		}
		targetID = created.TargetID
	}
	raw, err = c.call(ctx, "Target.attachToTarget", map[string]any{
		"targetId": targetID,
		"flatten":  true,
	}, "")
	if err != nil {
		return "", err
	}
	var attached struct {
		SessionID string `json:"sessionId"`
	}
	if err := json.Unmarshal(raw, &attached); err != nil {
		return "", fmt.Errorf("parse attached target: %w", err)
	}
	if attached.SessionID == "" {
		return "", errors.New("attach target returned empty sessionId")
	}
	c.pageSessionID = attached.SessionID
	return attached.SessionID, nil
}

func unmarshalRuntimeValue(raw json.RawMessage, op string, out any) error {
	var eval runtimeEvaluateResult
	if err := json.Unmarshal(raw, &eval); err != nil {
		return fmt.Errorf("parse %s result: %w", op, err)
	}
	if eval.ExceptionDetails != nil {
		return fmt.Errorf("%s JavaScript exception: %s", op, eval.ExceptionDetails.message())
	}
	if err := json.Unmarshal(eval.Result.Value, out); err != nil {
		return fmt.Errorf("parse %s value: %w", op, err)
	}
	return nil
}

func (d *runtimeExceptionDetails) message() string {
	if d == nil {
		return "unknown exception"
	}
	if d.Exception != nil && strings.TrimSpace(d.Exception.Description) != "" {
		return strings.TrimSpace(d.Exception.Description)
	}
	if strings.TrimSpace(d.Text) != "" {
		return strings.TrimSpace(d.Text)
	}
	return "unknown exception"
}

func (c *cdpConn) call(ctx context.Context, method string, params any, sessionID string) (json.RawMessage, error) {
	c.nextID++
	id := c.nextID
	msg := map[string]any{
		"id":         id,
		cdpKeyMethod: method,
	}
	if params != nil {
		msg["params"] = params
	}
	if sessionID != "" {
		msg["sessionId"] = sessionID
	}
	if err := c.conn.SetWriteDeadline(deadline(ctx)); err != nil {
		return nil, err
	}
	if err := c.conn.WriteJSON(msg); err != nil {
		return nil, fmt.Errorf("cdp %s: %w", method, err)
	}
	for {
		if resp, ok := c.takeResponse(id); ok {
			return c.callResult(method, resp)
		}
		resp, err := c.readMessage(ctx)
		if err != nil {
			return nil, fmt.Errorf("cdp %s: %w", method, err)
		}
		if handled, err := c.handleInterceptedEvent(ctx, resp); handled {
			if err != nil {
				return nil, err
			}
			continue
		}
		if resp.ID != id {
			c.bufferMessage(resp)
			continue
		}
		return c.callResult(method, resp)
	}
}

func (c *cdpConn) waitEvent(ctx context.Context, sessionID, method string) error {
	if c.takeEvent(sessionID, method) {
		return nil
	}
	for {
		msg, err := c.readMessage(ctx)
		if err != nil {
			return fmt.Errorf("wait for %s: %w", method, err)
		}
		if handled, err := c.handleInterceptedEvent(ctx, msg); handled {
			if err != nil {
				return err
			}
			continue
		}
		if msg.Method == method && (sessionID == "" || msg.SessionID == sessionID) {
			return nil
		}
		c.bufferMessage(msg)
	}
}

func (c *cdpConn) handleInterceptedEvent(ctx context.Context, msg cdpMessage) (bool, error) {
	switch msg.Method {
	case cdpEventRequestPaused:
		return c.handlePausedRequest(ctx, msg)
	case cdpEventAuthRequired:
		return c.handleAuthRequired(ctx, msg)
	default:
		return false, nil
	}
}

func (c *cdpConn) handlePausedRequest(ctx context.Context, msg cdpMessage) (bool, error) {
	if c.requestHandler == nil {
		return false, nil
	}
	var paused pausedRequest
	if err := json.Unmarshal(msg.Params, &paused); err != nil {
		return true, fmt.Errorf("parse paused browser request: %w", err)
	}
	sessionID := msg.SessionID
	if sessionID == "" {
		sessionID = c.pageSessionID
	}
	if paused.RequestID == "" {
		return true, errors.New("paused browser request has no requestId")
	}
	request := newBrowserRequest(paused)
	original := cloneBrowserRequest(request)
	response, err := c.requestHandler(ctx, request)
	if err != nil {
		return true, c.failPausedRequest(ctx, sessionID, paused.RequestID, err)
	}
	if response != nil {
		return true, c.fulfillPausedRequest(ctx, sessionID, paused.RequestID, response)
	}
	return true, c.continuePausedRequest(ctx, sessionID, paused.RequestID, original, request)
}

func (c *cdpConn) handleAuthRequired(ctx context.Context, msg cdpMessage) (bool, error) {
	if c.authHandler == nil {
		return false, nil
	}
	var required authRequired
	if err := json.Unmarshal(msg.Params, &required); err != nil {
		return true, fmt.Errorf("parse browser authentication challenge: %w", err)
	}
	if required.RequestID == "" {
		return true, errors.New("browser authentication challenge has no requestId")
	}
	sessionID := msg.SessionID
	if sessionID == "" {
		sessionID = c.pageSessionID
	}
	challenge := AuthChallenge{
		Request: BrowserRequest{
			URL:          required.Request.URL,
			Method:       required.Request.Method,
			Headers:      stringMapHeader(required.Request.Headers),
			ResourceType: required.ResourceType,
			FrameID:      required.FrameID,
		},
		Source: required.AuthChallenge.Source,
		Origin: required.AuthChallenge.Origin,
		Scheme: required.AuthChallenge.Scheme,
		Realm:  required.AuthChallenge.Realm,
	}
	if required.Request.PostData != nil {
		challenge.Request.PostData = []byte(*required.Request.PostData)
	}
	response, err := c.authHandler(ctx, challenge)
	if err != nil {
		return true, c.cancelAuthChallenge(
			ctx,
			sessionID,
			required.RequestID,
			fmt.Errorf("authentication handler: %w", err),
		)
	}
	response, err = normalizeAuthResponse(response)
	if err != nil {
		return true, c.cancelAuthChallenge(ctx, sessionID, required.RequestID, err)
	}
	return true, c.continueWithAuth(ctx, sessionID, required.RequestID, response)
}

func stringMapHeader(values map[string]string) http.Header {
	headers := make(http.Header, len(values))
	for name, value := range values {
		headers.Set(name, value)
	}
	return headers
}

func normalizeAuthResponse(response AuthResponse) (AuthResponse, error) {
	if response.Action == "" {
		response.Action = AuthActionDefault
	}
	switch response.Action {
	case AuthActionDefault, AuthActionCancel:
		if response.Username != "" || response.Password != "" {
			return AuthResponse{}, fmt.Errorf(
				"authentication action %q cannot include credentials",
				response.Action,
			)
		}
	case AuthActionProvideCredentials:
	default:
		return AuthResponse{}, fmt.Errorf("invalid authentication action %q", response.Action)
	}
	return response, nil
}

func (c *cdpConn) cancelAuthChallenge(
	ctx context.Context,
	sessionID string,
	requestID string,
	cause error,
) error {
	cancelErr := c.continueWithAuth(ctx, sessionID, requestID, AuthResponse{Action: AuthActionCancel})
	if cancelErr != nil {
		return errors.Join(cause, cancelErr)
	}
	return cause
}

func (c *cdpConn) continueWithAuth(
	ctx context.Context,
	sessionID string,
	requestID string,
	response AuthResponse,
) error {
	protocolAction := cdpAuthDefault
	switch response.Action {
	case AuthActionDefault:
	case AuthActionCancel:
		protocolAction = cdpAuthCancel
	case AuthActionProvideCredentials:
		protocolAction = cdpAuthCredentials
	default:
		return fmt.Errorf("invalid authentication action %q", response.Action)
	}
	authResponse := map[string]any{"response": protocolAction}
	if response.Action == AuthActionProvideCredentials {
		authResponse["username"] = response.Username
		authResponse["password"] = response.Password
	}
	_, err := c.call(ctx, "Fetch.continueWithAuth", map[string]any{
		cdpKeyRequestID:         requestID,
		"authChallengeResponse": authResponse,
	}, sessionID)
	return err
}

func newBrowserRequest(paused pausedRequest) *BrowserRequest {
	headers := make(http.Header, len(paused.Request.Headers))
	for name, value := range paused.Request.Headers {
		headers.Set(name, value)
	}
	var postData []byte
	if paused.Request.PostData != nil {
		postData = []byte(*paused.Request.PostData)
	}
	return &BrowserRequest{
		URL:                 paused.Request.URL,
		Method:              paused.Request.Method,
		Headers:             headers,
		PostData:            postData,
		ResourceType:        paused.ResourceType,
		FrameID:             paused.FrameID,
		NetworkID:           paused.NetworkID,
		RedirectedRequestID: paused.RedirectedRequestID,
	}
}

func cloneBrowserRequest(request *BrowserRequest) *BrowserRequest {
	cloned := *request
	cloned.Headers = request.Headers.Clone()
	cloned.PostData = slices.Clone(request.PostData)
	return &cloned
}

func (c *cdpConn) continuePausedRequest(
	ctx context.Context,
	sessionID string,
	requestID string,
	original *BrowserRequest,
	request *BrowserRequest,
) error {
	params := map[string]any{cdpKeyRequestID: requestID}
	if request.URL != original.URL {
		params[paramURL] = request.URL
	}
	if request.Method != original.Method {
		params["method"] = request.Method
	}
	if !headersEqual(request.Headers, original.Headers) {
		params["headers"] = headerEntries(request.Headers)
	}
	if (request.PostData == nil) != (original.PostData == nil) ||
		!bytes.Equal(request.PostData, original.PostData) {
		// CDP defines Fetch.continueRequest.postData as base64 when passed over JSON.
		params["postData"] = base64.StdEncoding.EncodeToString(request.PostData)
	}
	_, err := c.call(ctx, "Fetch.continueRequest", params, sessionID)
	return err
}

func (c *cdpConn) fulfillPausedRequest(
	ctx context.Context,
	sessionID string,
	requestID string,
	response *BrowserResponse,
) error {
	statusCode := response.StatusCode
	if statusCode == 0 {
		statusCode = http.StatusOK
	}
	if statusCode < 100 || statusCode > 599 {
		return c.failPausedRequest(ctx, sessionID, requestID, fmt.Errorf(
			"request middleware returned invalid response status %d",
			statusCode,
		))
	}
	params := map[string]any{
		cdpKeyRequestID: requestID,
		"responseCode":  statusCode,
	}
	if response.StatusText != "" {
		params["responsePhrase"] = response.StatusText
	}
	if response.Headers != nil {
		params["responseHeaders"] = headerEntries(response.Headers)
	}
	if response.Body != nil {
		params["body"] = base64.StdEncoding.EncodeToString(response.Body)
	}
	_, err := c.call(ctx, "Fetch.fulfillRequest", params, sessionID)
	return err
}

func (c *cdpConn) failPausedRequest(
	ctx context.Context,
	sessionID string,
	requestID string,
	cause error,
) error {
	_, failErr := c.call(ctx, "Fetch.failRequest", map[string]any{
		cdpKeyRequestID: requestID,
		"errorReason":   "BlockedByClient",
	}, sessionID)
	requestErr := fmt.Errorf("browser request: %w", cause)
	if failErr != nil {
		return errors.Join(requestErr, failErr)
	}
	return requestErr
}

func headersEqual(a, b http.Header) bool {
	return maps.EqualFunc(a, b, slices.Equal)
}

func headerEntries(headers http.Header) []map[string]string {
	names := make([]string, 0, len(headers))
	for name := range headers {
		names = append(names, name)
	}
	sort.Strings(names)
	entries := make([]map[string]string, 0, len(headers))
	for _, name := range names {
		for _, value := range headers[name] {
			entries = append(entries, map[string]string{"name": name, "value": value})
		}
	}
	return entries
}

func (c *cdpConn) callResult(method string, resp cdpMessage) (json.RawMessage, error) {
	if resp.Error != nil {
		return nil, fmt.Errorf("cdp %s: %s", method, resp.Error.Message)
	}
	return resp.Result, nil
}

func (c *cdpConn) readMessage(ctx context.Context) (cdpMessage, error) {
	if err := c.conn.SetReadDeadline(deadline(ctx)); err != nil {
		return cdpMessage{}, err
	}
	var msg cdpMessage
	if err := c.conn.ReadJSON(&msg); err != nil {
		return cdpMessage{}, err
	}
	return msg, nil
}

func (c *cdpConn) bufferMessage(msg cdpMessage) {
	if msg.ID != 0 {
		if c.responses == nil {
			c.responses = map[int64]cdpMessage{}
		}
		c.responses[msg.ID] = msg
		return
	}
	c.events = append(c.events, msg)
}

func (c *cdpConn) takeResponse(id int64) (cdpMessage, bool) {
	if c.responses == nil {
		return cdpMessage{}, false
	}
	msg, ok := c.responses[id]
	if ok {
		delete(c.responses, id)
	}
	return msg, ok
}

func (c *cdpConn) takeEvent(sessionID, method string) bool {
	for i, msg := range c.events {
		if msg.Method == method && (sessionID == "" || msg.SessionID == sessionID) {
			c.events = append(c.events[:i], c.events[i+1:]...)
			return true
		}
	}
	return false
}

func deadline(ctx context.Context) time.Time {
	if d, ok := ctx.Deadline(); ok {
		return d
	}
	return time.Now().Add(defaultNavigationTimeout)
}
