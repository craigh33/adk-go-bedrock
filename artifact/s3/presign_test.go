package s3artifact

import (
	"context"
	"strings"
	"testing"
	"time"
)

// fakePresigner records calls and returns a deterministic URL.
type fakePresigner struct {
	calls []fakePresignCall
	err   error
}

type fakePresignCall struct {
	bucket string
	key    string
	ttl    time.Duration
}

func (f *fakePresigner) PresignGetObject(_ context.Context, bucket, key string, ttl time.Duration) (string, error) {
	f.calls = append(f.calls, fakePresignCall{bucket: bucket, key: key, ttl: ttl})
	if f.err != nil {
		return "", f.err
	}
	return "https://s3.example.com/" + bucket + "/" + key + "?X-Amz-Signature=fake", nil
}

func TestPresignLoadReturnsURL(t *testing.T) {
	fake := newFakeS3()
	presigner := &fakePresigner{}
	svc := newTestService(t, fake, Config{Bucket: "b", Presigner: presigner})
	saveText(t, svc, "report.txt", "hello")

	resp, err := svc.PresignLoad(context.Background(), &PresignLoadRequest{
		AppName: "app", UserID: "u1", SessionID: "s1", FileName: "report.txt",
	})
	if err != nil {
		t.Fatalf("PresignLoad: %v", err)
	}
	if resp.Version != 1 {
		t.Fatalf("Version = %d, want 1", resp.Version)
	}
	if !strings.Contains(resp.URL, "app/u1/s1/report.txt/1") {
		t.Fatalf("URL = %q, want to contain object key", resp.URL)
	}
	if resp.Expires.IsZero() {
		t.Fatal("Expires not set")
	}
}

func TestPresignLoadResolvesLatestVersion(t *testing.T) {
	fake := newFakeS3()
	presigner := &fakePresigner{}
	svc := newTestService(t, fake, Config{Bucket: "b", Presigner: presigner})
	saveText(t, svc, "report.txt", "v1")
	saveText(t, svc, "report.txt", "v2")

	resp, err := svc.PresignLoad(context.Background(), &PresignLoadRequest{
		AppName: "app", UserID: "u1", SessionID: "s1", FileName: "report.txt",
	})
	if err != nil {
		t.Fatalf("PresignLoad: %v", err)
	}
	if resp.Version != 2 {
		t.Fatalf("Version = %d, want 2", resp.Version)
	}
	if !strings.HasSuffix(presigner.calls[0].key, "/2") {
		t.Fatalf("key = %q, want to end in /2", presigner.calls[0].key)
	}
}

func TestPresignLoadSpecificVersion(t *testing.T) {
	fake := newFakeS3()
	presigner := &fakePresigner{}
	svc := newTestService(t, fake, Config{Bucket: "b", Presigner: presigner})
	saveText(t, svc, "report.txt", "v1")
	saveText(t, svc, "report.txt", "v2")

	resp, err := svc.PresignLoad(context.Background(), &PresignLoadRequest{
		AppName: "app", UserID: "u1", SessionID: "s1", FileName: "report.txt", Version: 1,
	})
	if err != nil {
		t.Fatalf("PresignLoad v1: %v", err)
	}
	if resp.Version != 1 {
		t.Fatalf("Version = %d, want 1", resp.Version)
	}
}

func TestPresignLoadUsesRequestTTL(t *testing.T) {
	fake := newFakeS3()
	presigner := &fakePresigner{}
	svc := newTestService(t, fake, Config{Bucket: "b", Presigner: presigner, PresignTTL: time.Hour})
	saveText(t, svc, "report.txt", "x")

	want := 5 * time.Minute
	if _, err := svc.PresignLoad(context.Background(), &PresignLoadRequest{
		AppName: "app", UserID: "u1", SessionID: "s1", FileName: "report.txt",
		TTL: want,
	}); err != nil {
		t.Fatalf("PresignLoad: %v", err)
	}
	if got := presigner.calls[0].ttl; got != want {
		t.Fatalf("ttl = %v, want %v", got, want)
	}
}

func TestPresignLoadUsesConfigTTL(t *testing.T) {
	fake := newFakeS3()
	presigner := &fakePresigner{}
	want := 2 * time.Hour
	svc := newTestService(t, fake, Config{Bucket: "b", Presigner: presigner, PresignTTL: want})
	saveText(t, svc, "report.txt", "x")

	if _, err := svc.PresignLoad(context.Background(), &PresignLoadRequest{
		AppName: "app", UserID: "u1", SessionID: "s1", FileName: "report.txt",
	}); err != nil {
		t.Fatalf("PresignLoad: %v", err)
	}
	if got := presigner.calls[0].ttl; got != want {
		t.Fatalf("ttl = %v, want %v (Config.PresignTTL)", got, want)
	}
}

func TestPresignLoadDefaultsTTLTo15Minutes(t *testing.T) {
	fake := newFakeS3()
	presigner := &fakePresigner{}
	svc := newTestService(t, fake, Config{Bucket: "b", Presigner: presigner})
	saveText(t, svc, "report.txt", "x")

	if _, err := svc.PresignLoad(context.Background(), &PresignLoadRequest{
		AppName: "app", UserID: "u1", SessionID: "s1", FileName: "report.txt",
	}); err != nil {
		t.Fatalf("PresignLoad: %v", err)
	}
	if got := presigner.calls[0].ttl; got != defaultPresignTTL {
		t.Fatalf("ttl = %v, want %v (default)", got, defaultPresignTTL)
	}
}

func TestPresignLoadErrorsWithoutPresigner(t *testing.T) {
	svc := newTestService(t, newFakeS3(), Config{Bucket: "b"})
	_, err := svc.PresignLoad(context.Background(), &PresignLoadRequest{
		AppName: "app", UserID: "u1", SessionID: "s1", FileName: "report.txt",
	})
	if err == nil {
		t.Fatal("want error when Presigner not configured")
	}
}

func TestPresignLoadErrorsWhenArtifactNotFound(t *testing.T) {
	presigner := &fakePresigner{}
	svc := newTestService(t, newFakeS3(), Config{Bucket: "b", Presigner: presigner})
	_, err := svc.PresignLoad(context.Background(), &PresignLoadRequest{
		AppName: "app", UserID: "u1", SessionID: "s1", FileName: "nope.txt",
	})
	if !isFSNotExist(err) {
		t.Fatalf("want fs.ErrNotExist, got %v", err)
	}
}

func TestPresignLoadPassesBucketToPresigner(t *testing.T) {
	fake := newFakeS3()
	presigner := &fakePresigner{}
	svc := newTestService(t, fake, Config{Bucket: "my-bucket", Presigner: presigner})
	saveText(t, svc, "f.txt", "x")

	if _, err := svc.PresignLoad(context.Background(), &PresignLoadRequest{
		AppName: "app", UserID: "u1", SessionID: "s1", FileName: "f.txt",
	}); err != nil {
		t.Fatalf("PresignLoad: %v", err)
	}
	if presigner.calls[0].bucket != "my-bucket" {
		t.Fatalf("bucket = %q, want my-bucket", presigner.calls[0].bucket)
	}
}
