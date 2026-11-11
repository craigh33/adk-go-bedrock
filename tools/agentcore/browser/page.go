package browser

import (
	"context"
	"errors"
	"strings"

	"github.com/aws/aws-sdk-go-v2/service/bedrockagentcore/types"
	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"

	browsermappers "github.com/craigh33/adk-go-bedrock/internal/agentcore/mappers/browser"
)

type browsePageArgs struct {
	URL        string `json:"url"                  jsonschema:"HTTP or HTTPS URL to open."`
	Selector   string `json:"selector,omitempty"   jsonschema:"Optional CSS selector to wait for and read. Defaults to the document body."`
	Screenshot string `json:"screenshot,omitempty" jsonschema:"Optional artifact filename for a full-page screenshot, such as page.png or page.jpg."`
}

type browsePageResult struct {
	Status     string              `json:"status"`
	URL        string              `json:"url"`
	Title      string              `json:"title"`
	Text       string              `json:"text"`
	Truncated  bool                `json:"truncated"`
	Screenshot *screenshotArtifact `json:"screenshot,omitempty"`
}

type screenshotArtifact struct {
	FileName string `json:"file_name"`
	Version  int64  `json:"version"`
	MIMEType string `json:"mime_type"`
}

// NewPage creates a browse_page tool that reads a page and optionally saves a screenshot in a single session.
func NewPage(cfg Config) (tool.Tool, error) {
	t, err := newBrowserTool(cfg)
	if err != nil {
		return nil, err
	}
	return functiontool.New(functiontool.Config{
		Name: "browse_page",
		Description: "Open a URL and return the visible text, page title, and final URL. " +
			"Supply a screenshot filename to also save a full-page screenshot. " +
			"The tool opens and stops its Browser session within the call.",
	}, t.runPage)
}

//nolint:nonamedreturns // The deferred stop adds any cleanup error to the operation error.
func (t *browserTool) runPage(ctx agent.Context, args browsePageArgs) (result browsePageResult, err error) {
	args.URL = strings.TrimSpace(args.URL)
	if err := t.checkURL(ctx, args.URL, URLStageNavigate); err != nil {
		return browsePageResult{}, err
	}
	args.Selector = strings.TrimSpace(args.Selector)
	var file screenshotFile
	var artifacts agent.Artifacts
	if name := strings.TrimSpace(args.Screenshot); name != "" {
		file, err = resolveScreenshotFile(name, "")
		if err != nil {
			return browsePageResult{}, err
		}
		artifacts = ctx.Artifacts()
		if artifacts == nil {
			return browsePageResult{}, errors.New("artifact service is unavailable")
		}
	}
	pageCtx, cancel := context.WithTimeout(ctx, t.navigationTimeout)
	defer cancel()
	var session browserSession
	started, err := t.startSession(pageCtx, ctx, &session)
	if err != nil {
		return browsePageResult{}, err
	}
	defer func() {
		err = t.cleanupStartedSession(ctx, &session, err)
	}()
	return t.visitPage(pageCtx, started.Streams, args, file, artifacts)
}

func (t *browserTool) visitPage(
	ctx context.Context,
	streams *types.BrowserSessionStream,
	args browsePageArgs,
	file screenshotFile,
	artifacts agent.Artifacts,
) (browsePageResult, error) {
	cdp, err := t.openCDP(ctx, browsermappers.AgentCoreBrowserAutomationEndpoint(streams))
	if err != nil {
		return browsePageResult{}, err
	}
	defer cdp.close()
	if _, err := t.navigatePage(ctx, cdp, args.URL, t.waitUntil, args.Selector); err != nil {
		return browsePageResult{}, err
	}
	text, err := t.extractPageText(ctx, cdp, args.Selector)
	if err != nil {
		return browsePageResult{}, err
	}
	result := browsePageResult{
		Status:    statusSuccess,
		URL:       text.URL,
		Title:     text.Title,
		Text:      text.Text,
		Truncated: text.Truncated,
	}
	if file.name != "" {
		_, version, err := t.savePageScreenshot(ctx, cdp, artifacts, file, true, nil)
		if err != nil {
			return browsePageResult{}, err
		}
		result.Screenshot = &screenshotArtifact{FileName: file.name, Version: version, MIMEType: file.mimeType}
	}
	return result, nil
}
