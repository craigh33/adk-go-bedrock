package mappers

import (
	"testing"
)

func TestMIMETypeFromExtension(t *testing.T) {
	t.Parallel()
	tests := []struct {
		path string
		want string
	}{
		{"report.PDF", mimeApplicationPDF},
		{"/tmp/x.docx", mimeApplicationDOCX},
		{"slides.pptx", mimeApplicationPPTX},
		{"a.png", mimeImagePNG},
		{"b.JPEG", mimeImageJPEG},
		{"clip.mp4", mimeVideoMP4},
		{"data.json", mimeApplicationJSON},
		{"trace.har", mimeApplicationJSON},
		{"out.log", mimeTextPlain},
		{"k8s.yaml", mimeTextYAML},
		{"app.xml", mimeApplicationXML},
		{"cfg.toml", mimeApplicationTOML},
		{"main.go", mimeTextPlain},
		{"lib.rs", mimeTextPlain},
		{"unknown.xyz", mimeApplicationOctetStream},
		{"noext", mimeApplicationOctetStream},
	}
	for _, tt := range tests {
		if got := MIMETypeFromExtension(tt.path); got != tt.want {
			t.Errorf("MIMETypeFromExtension(%q) = %q, want %q", tt.path, got, tt.want)
		}
	}
}

func TestInferDocumentMIMEFromFilename_usesMIMETypeFromExtension(t *testing.T) {
	t.Parallel()
	m, ok := inferDocumentMIMEFromFilename("/home/u/f/memo.docx")
	if !ok || m != mimeApplicationDOCX {
		t.Fatalf("got %q, ok=%v", m, ok)
	}
	_, ok = inferDocumentMIMEFromFilename("photo.png")
	if ok {
		t.Fatal("expected no document inference for .png")
	}
	m, ok = inferDocumentMIMEFromFilename("/tmp/a/data.json")
	if !ok || m != mimeApplicationJSON {
		t.Fatalf("infer .json: got %q, ok=%v", m, ok)
	}
	m, ok = inferDocumentMIMEFromFilename("errors.log")
	if !ok || m != mimeTextPlain {
		t.Fatalf("infer .log: got %q, ok=%v", m, ok)
	}
}
