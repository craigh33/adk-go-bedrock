package browser

import "time"

const (
	// ToolName is the name exposed to ADK agents.
	ToolName = "agentcore_browser"

	defaultBrowserIdentifier  = "aws.browser.v1"
	defaultSessionTimeout     = int32(900)
	minSessionTimeout         = int32(60)
	maxSessionTimeout         = int32(28_800)
	defaultNavigationTimeout  = 30 * time.Second
	defaultCleanupTimeout     = 10 * time.Second
	defaultMaxTextBytes       = 64 << 10
	defaultMaxScreenshotBytes = int64(16 << 20)

	paramAction          = "action"
	paramSessionID       = "session_id"
	paramURL             = "url"
	paramSelector        = "selector"
	paramFileName        = "file_name"
	paramFormat          = "format"
	paramWaitUntil       = "wait_until"
	paramWaitForSelector = "wait_for_selector"
	paramFullPage        = "full_page"
	paramQuality         = "quality"

	actionStart       = "start"
	actionNavigate    = "navigate"
	actionExtractText = "extract_text"
	actionScreenshot  = "screenshot"
	actionStatus      = "status"
	actionStop        = "stop"

	schemaTypeString       = "STRING"
	schemaTypeBoolean      = "BOOLEAN"
	schemaTypeInteger      = "INTEGER"
	schemaFormatEnum       = "enum"
	resultKeyStatus        = actionStatus
	resultKeyBrowserID     = "browser_identifier"
	resultKeySessionStatus = "session_status"
	resultKeyTitle         = "title"
	screenshotFormatPNG    = "png"
	screenshotFormatJPEG   = "jpeg"
	screenshotFormatJPG    = "jpg"
	mimeTypeJPEG           = "image/jpeg"
	cdpKeyMethod           = "method"
	cdpKeyRequestID        = "requestId"
	cdpKeyExpression       = "expression"
	cdpKeyReturnByValue    = "returnByValue"
	cdpEventRequestPaused  = "Fetch.requestPaused"
	cdpEventAuthRequired   = "Fetch.authRequired"
	schemeHTTP             = "http"
	schemeHTTPS            = "https"
	schemeData             = "data"
	schemeBlob             = "blob"
	cdpAuthDefault         = "Default"
	cdpAuthCancel          = "CancelAuth"
	cdpAuthCredentials     = "ProvideCredentials"

	statusSuccess      = "success"
	statusNoSession    = "NO_SESSION"
	serviceID          = "bedrock-agentcore"
	emptyPayloadSHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)
