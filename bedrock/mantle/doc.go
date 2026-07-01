// Package mantle provides an Amazon Bedrock Mantle implementation of the
// [github.com/craigh33/adk-go-bedrock/bedrock/converse.RuntimeAPI] interface.
//
// Bedrock Mantle exposes an Anthropic-compatible Messages API as an alternative
// transport to the native Bedrock Converse API. This package wraps
// [github.com/anthropics/anthropic-sdk-go/bedrock.MantleClient] behind the same
// RuntimeAPI interface used by the Converse path, translating Converse request
// and response shapes to and from the Anthropic Messages API. This lets a
// [github.com/craigh33/adk-go-bedrock/bedrock/converse.Model] target Mantle
// without any changes to the model, the genai translation layer, or the
// streaming assembly machinery.
//
// It lives in its own Go module so the anthropic-sdk-go dependency and Mantle's
// faster-moving API surface stay isolated from the core Converse module's
// go.sum and release cadence. The modules are tied together via the repository
// root go.work workspace file.
package mantle
