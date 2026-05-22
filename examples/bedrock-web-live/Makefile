.PHONY: run

# `live` must be in the sublauncher keyword list — without it, web.Sublauncher
# treats our /run_live route as inactive and falls through to the upstream
# Gemini-only handler (which fails with "model does not support live connection").
run:
	go run . web api webui live
