package mantle

import "testing"

func TestNormalizeModelID(t *testing.T) {
	cases := map[string]string{
		"anthropic.claude-sonnet-4-5-20250929-v1:0":    "anthropic.claude-sonnet-4-5-20250929-v1:0",
		"us.anthropic.claude-sonnet-4-5-20250929-v1:0": "anthropic.claude-sonnet-4-5-20250929-v1:0",
		"eu.anthropic.claude-3-5-sonnet-20241022-v2:0": "anthropic.claude-3-5-sonnet-20241022-v2:0",
		"apac.anthropic.claude-3-haiku":                "anthropic.claude-3-haiku",
		"global.anthropic.claude-opus-4":               "anthropic.claude-opus-4",
		"us-gov.anthropic.claude-3-haiku":              "anthropic.claude-3-haiku",
		"  us.anthropic.claude-3-haiku  ":              "anthropic.claude-3-haiku",
		"anthropic.claude-3-haiku":                     "anthropic.claude-3-haiku",
	}
	for in, want := range cases {
		if got := NormalizeModelID(in); got != want {
			t.Errorf("NormalizeModelID(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestValidateModelID(t *testing.T) {
	valid := []string{
		"anthropic.claude-sonnet-4-5-20250929-v1:0",
		"us.anthropic.claude-3-5-sonnet-20241022-v2:0",
		"global.anthropic.claude-opus-4",
	}
	for _, id := range valid {
		if err := ValidateModelID(id); err != nil {
			t.Errorf("ValidateModelID(%q) unexpected error: %v", id, err)
		}
	}

	invalid := []string{
		"",
		"   ",
		"amazon.nova-2-lite-v1:0",
		"us.amazon.nova-2-lite-v1:0",
		"meta.llama3-70b",
	}
	for _, id := range invalid {
		if err := ValidateModelID(id); err == nil {
			t.Errorf("ValidateModelID(%q) = nil, want error", id)
		}
	}
}
