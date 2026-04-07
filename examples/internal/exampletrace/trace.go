// Package exampletrace provides a small OpenTelemetry setup for examples.
package exampletrace

import (
	"context"
	"fmt"
	"os"

	"go.opentelemetry.io/otel/exporters/stdout/stdouttrace"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

// TracerProvider builds an SDK [sdktrace.TracerProvider] that writes trace spans to
// stdout as JSON (pretty-printed). Call the returned shutdown before exit so the batch
// processor flushes.
func TracerProvider(ctx context.Context) (*sdktrace.TracerProvider, func(context.Context) error, error) {
	_ = ctx // reserved for future resource detection
	exp, err := stdouttrace.New(
		stdouttrace.WithPrettyPrint(),
		stdouttrace.WithWriter(os.Stdout),
	)
	if err != nil {
		return nil, nil, fmt.Errorf("stdout trace exporter: %w", err)
	}
	tp := sdktrace.NewTracerProvider(sdktrace.WithBatcher(exp))
	return tp, tp.Shutdown, nil
}
