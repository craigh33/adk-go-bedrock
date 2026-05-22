package webbridge

import (
	"flag"
	"fmt"
	"net/http"

	"github.com/gorilla/mux"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/cmd/launcher/web"
)

// SublauncherOptions configures [NewSublauncher].
type SublauncherOptions struct {
	// Keyword is the CLI keyword that activates this sublauncher within the
	// web launcher. Empty defaults to "live". Users invoke e.g.
	// `go run . web api webui live` to mount it.
	Keyword string

	// Path is where the handler mounts on the launcher's router. Empty
	// defaults to "/api/run_live", matching the prefix the upstream API
	// sublauncher serves at and the path adk-go's UI is hardcoded to.
	Path string

	// Description is what `--help` shows for this sublauncher.
	// Empty defaults to "Nova Sonic bidirectional voice".
	Description string
}

// NewSublauncher wraps an [http.Handler] (typically the one returned by
// [New]) as an adk-go [web.Sublauncher] so it can be mounted alongside
// webui, api, etc. via [web.NewLauncher].
//
// Register this BEFORE web.api.NewLauncher() in your composition so the
// exact-path route registered by SetupSubrouters wins over the upstream
// API sublauncher's catchall.
func NewSublauncher(handler http.Handler, opts SublauncherOptions) web.Sublauncher {
	if opts.Keyword == "" {
		opts.Keyword = "live"
	}
	if opts.Path == "" {
		opts.Path = "/api/run_live"
	}
	if opts.Description == "" {
		opts.Description = "Nova Sonic bidirectional voice"
	}
	return &sublauncher{
		handler: handler,
		opts:    opts,
		flags:   flag.NewFlagSet(opts.Keyword, flag.ContinueOnError),
	}
}

type sublauncher struct {
	handler http.Handler
	opts    SublauncherOptions
	flags   *flag.FlagSet
}

func (s *sublauncher) Keyword() string           { return s.opts.Keyword }
func (s *sublauncher) SimpleDescription() string { return s.opts.Description }
func (s *sublauncher) CommandLineSyntax() string { return "    (no flags)" }
func (s *sublauncher) Parse(args []string) ([]string, error) {
	return args, nil
}

func (s *sublauncher) UserMessage(webURL string, printer func(v ...any)) {
	printer(fmt.Sprintf("      %s:  Nova Sonic voice bridge active at %s%s",
		s.opts.Keyword, webURL, s.opts.Path))
}

func (s *sublauncher) SetupSubrouters(router *mux.Router, _ *launcher.Config) error {
	router.Methods(http.MethodGet, http.MethodOptions).Path(s.opts.Path).Handler(s.handler)
	return nil
}
