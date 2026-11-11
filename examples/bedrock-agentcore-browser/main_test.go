package main

import (
	"slices"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
)

func TestBrowserConfigAllowedHosts(t *testing.T) {
	for _, tc := range []struct {
		name  string
		value string
		want  []string
	}{
		{name: "default", want: []string{"example.com"}},
		{name: "custom", value: " www.example.org, static.example.org ", want: []string{"www.example.org", "static.example.org"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("AGENTCORE_BROWSER_ALLOWED_HOSTS", tc.value)
			cfg, err := browserConfigFromEnv(aws.Config{Region: "us-east-1"})
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(cfg.AllowedHosts, tc.want) {
				t.Fatalf("allowed hosts = %v, want %v", cfg.AllowedHosts, tc.want)
			}
		})
	}
}
