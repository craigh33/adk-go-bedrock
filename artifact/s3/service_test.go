package s3artifact

import (
	"bytes"
	"context"
	"errors"
	"io"
	"io/fs"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/aws-sdk-go-v2/service/s3/types"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/artifact"
)

// fakeS3 is an in-memory Client, storing objects by key.
type fakeS3 struct {
	objects map[string]fakeObject
	// listPageSize forces ListObjectsV2 pagination when > 0.
	listPageSize int
	// putSSE records the encryption settings of the last PutObject.
	putSSE      types.ServerSideEncryption
	putSSEKeyID string
}

type fakeObject struct {
	data         []byte
	contentType  string
	lastModified time.Time
}

func newFakeS3() *fakeS3 {
	return &fakeS3{objects: map[string]fakeObject{}}
}

func (f *fakeS3) PutObject(_ context.Context, in *s3.PutObjectInput, _ ...func(*s3.Options)) (*s3.PutObjectOutput, error) {
	data, err := io.ReadAll(in.Body)
	if err != nil {
		return nil, err
	}
	f.objects[aws.ToString(in.Key)] = fakeObject{
		data:         data,
		contentType:  aws.ToString(in.ContentType),
		lastModified: time.Unix(1700000000, 0),
	}
	f.putSSE = in.ServerSideEncryption
	f.putSSEKeyID = aws.ToString(in.SSEKMSKeyId)
	return &s3.PutObjectOutput{}, nil
}

func (f *fakeS3) GetObject(_ context.Context, in *s3.GetObjectInput, _ ...func(*s3.Options)) (*s3.GetObjectOutput, error) {
	obj, ok := f.objects[aws.ToString(in.Key)]
	if !ok {
		return nil, &types.NoSuchKey{}
	}
	return &s3.GetObjectOutput{
		Body:        io.NopCloser(bytes.NewReader(obj.data)),
		ContentType: aws.String(obj.contentType),
	}, nil
}

func (f *fakeS3) HeadObject(_ context.Context, in *s3.HeadObjectInput, _ ...func(*s3.Options)) (*s3.HeadObjectOutput, error) {
	obj, ok := f.objects[aws.ToString(in.Key)]
	if !ok {
		return nil, &types.NotFound{}
	}
	return &s3.HeadObjectOutput{
		ContentType:  aws.String(obj.contentType),
		LastModified: aws.Time(obj.lastModified),
		Metadata:     map[string]string{"origin": "test"},
	}, nil
}

func (f *fakeS3) DeleteObject(_ context.Context, in *s3.DeleteObjectInput, _ ...func(*s3.Options)) (*s3.DeleteObjectOutput, error) {
	delete(f.objects, aws.ToString(in.Key))
	return &s3.DeleteObjectOutput{}, nil
}

func (f *fakeS3) ListObjectsV2(_ context.Context, in *s3.ListObjectsV2Input, _ ...func(*s3.Options)) (*s3.ListObjectsV2Output, error) {
	var keys []string
	for k := range f.objects {
		if strings.HasPrefix(k, aws.ToString(in.Prefix)) {
			keys = append(keys, k)
		}
	}
	sort.Strings(keys)

	start := 0
	if tok := aws.ToString(in.ContinuationToken); tok != "" {
		for i, k := range keys {
			if k > tok {
				start = i
				break
			}
		}
	}
	end := len(keys)
	truncated := false
	if f.listPageSize > 0 && start+f.listPageSize < len(keys) {
		end = start + f.listPageSize
		truncated = true
	}

	out := &s3.ListObjectsV2Output{IsTruncated: aws.Bool(truncated)}
	for _, k := range keys[start:end] {
		out.Contents = append(out.Contents, types.Object{Key: aws.String(k)})
	}
	if truncated {
		out.NextContinuationToken = aws.String(keys[end-1])
	}
	return out, nil
}

func newTestService(t *testing.T, client Client, cfg Config) *Service {
	t.Helper()
	svc, err := NewService(client, cfg)
	if err != nil {
		t.Fatalf("NewService: %v", err)
	}
	return svc
}

func saveText(t *testing.T, svc *Service, fileName, text string) int64 {
	t.Helper()
	resp, err := svc.Save(context.Background(), &artifact.SaveRequest{
		AppName: "app", UserID: "u1", SessionID: "s1", FileName: fileName,
		Part: genai.NewPartFromText(text),
	})
	if err != nil {
		t.Fatalf("Save(%s): %v", fileName, err)
	}
	return resp.Version
}

func TestSaveAssignsIncrementingVersionsAndKeys(t *testing.T) {
	fake := newFakeS3()
	svc := newTestService(t, fake, Config{Bucket: "b", KeyPrefix: "adk-artifacts"})

	if v := saveText(t, svc, "report.txt", "one"); v != 1 {
		t.Fatalf("first version = %d, want 1", v)
	}
	if v := saveText(t, svc, "report.txt", "two"); v != 2 {
		t.Fatalf("second version = %d, want 2", v)
	}
	if _, ok := fake.objects["adk-artifacts/app/u1/s1/report.txt/2"]; !ok {
		t.Fatalf("expected prefixed session-scoped key, got %v", keys(fake))
	}
}

func TestUserNamespacedKeys(t *testing.T) {
	fake := newFakeS3()
	svc := newTestService(t, fake, Config{Bucket: "b"})

	saveText(t, svc, "user:prefs.txt", "dark mode")
	if _, ok := fake.objects["app/u1/user/user:prefs.txt/1"]; !ok {
		t.Fatalf("expected user-scoped key, got %v", keys(fake))
	}
}

func TestLoadLatestAndSpecificVersion(t *testing.T) {
	fake := newFakeS3()
	svc := newTestService(t, fake, Config{Bucket: "b"})
	saveText(t, svc, "report.txt", "one")
	saveText(t, svc, "report.txt", "two")

	latest, err := svc.Load(context.Background(), &artifact.LoadRequest{
		AppName: "app", UserID: "u1", SessionID: "s1", FileName: "report.txt",
	})
	if err != nil {
		t.Fatalf("Load latest: %v", err)
	}
	if got := string(latest.Part.InlineData.Data); got != "two" {
		t.Fatalf("latest = %q, want %q", got, "two")
	}
	if mime := latest.Part.InlineData.MIMEType; mime != "text/plain" {
		t.Fatalf("mime = %q, want text/plain", mime)
	}

	v1, err := svc.Load(context.Background(), &artifact.LoadRequest{
		AppName: "app", UserID: "u1", SessionID: "s1", FileName: "report.txt", Version: 1,
	})
	if err != nil {
		t.Fatalf("Load v1: %v", err)
	}
	if got := string(v1.Part.InlineData.Data); got != "one" {
		t.Fatalf("v1 = %q, want %q", got, "one")
	}
}

func TestLoadMissingArtifactIsErrNotExist(t *testing.T) {
	svc := newTestService(t, newFakeS3(), Config{Bucket: "b"})
	_, err := svc.Load(context.Background(), &artifact.LoadRequest{
		AppName: "app", UserID: "u1", SessionID: "s1", FileName: "nope.txt",
	})
	if !isFSNotExist(err) {
		t.Fatalf("want fs.ErrNotExist, got %v", err)
	}
}

func TestDeleteSpecificVersionExposesPreviousAsLatest(t *testing.T) {
	fake := newFakeS3()
	svc := newTestService(t, fake, Config{Bucket: "b"})
	saveText(t, svc, "report.txt", "one")
	saveText(t, svc, "report.txt", "two")

	if err := svc.Delete(context.Background(), &artifact.DeleteRequest{
		AppName: "app", UserID: "u1", SessionID: "s1", FileName: "report.txt", Version: 2,
	}); err != nil {
		t.Fatalf("Delete v2: %v", err)
	}

	latest, err := svc.Load(context.Background(), &artifact.LoadRequest{
		AppName: "app", UserID: "u1", SessionID: "s1", FileName: "report.txt",
	})
	if err != nil {
		t.Fatalf("Load latest after delete: %v", err)
	}
	if got := string(latest.Part.InlineData.Data); got != "one" {
		t.Fatalf("latest after delete = %q, want %q", got, "one")
	}
}

func TestDeleteVersionZeroDeletesAllVersions(t *testing.T) {
	fake := newFakeS3()
	svc := newTestService(t, fake, Config{Bucket: "b"})
	saveText(t, svc, "report.txt", "one")
	saveText(t, svc, "report.txt", "two")

	if err := svc.Delete(context.Background(), &artifact.DeleteRequest{
		AppName: "app", UserID: "u1", SessionID: "s1", FileName: "report.txt",
	}); err != nil {
		t.Fatalf("Delete all: %v", err)
	}
	if len(fake.objects) != 0 {
		t.Fatalf("expected no objects, got %v", keys(fake))
	}
}

func TestListUnionsSessionAndUserScopesAcrossPages(t *testing.T) {
	fake := newFakeS3()
	fake.listPageSize = 1 // force pagination
	svc := newTestService(t, fake, Config{Bucket: "b"})
	saveText(t, svc, "a.txt", "x")
	saveText(t, svc, "b.txt", "y")
	saveText(t, svc, "user:c.txt", "z")

	resp, err := svc.List(context.Background(), &artifact.ListRequest{
		AppName: "app", UserID: "u1", SessionID: "s1",
	})
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	want := []string{"a.txt", "b.txt", "user:c.txt"}
	if len(resp.FileNames) != len(want) {
		t.Fatalf("List = %v, want %v", resp.FileNames, want)
	}
	for i := range want {
		if resp.FileNames[i] != want[i] {
			t.Fatalf("List = %v, want %v", resp.FileNames, want)
		}
	}
}

func TestVersionsErrorsWhenEmpty(t *testing.T) {
	svc := newTestService(t, newFakeS3(), Config{Bucket: "b"})
	_, err := svc.Versions(context.Background(), &artifact.VersionsRequest{
		AppName: "app", UserID: "u1", SessionID: "s1", FileName: "nope.txt",
	})
	if !isFSNotExist(err) {
		t.Fatalf("want fs.ErrNotExist, got %v", err)
	}
}

func TestGetArtifactVersionMetadata(t *testing.T) {
	fake := newFakeS3()
	svc := newTestService(t, fake, Config{Bucket: "b"})
	saveText(t, svc, "report.txt", "one")

	resp, err := svc.GetArtifactVersion(context.Background(), &artifact.GetArtifactVersionRequest{
		AppName: "app", UserID: "u1", SessionID: "s1", FileName: "report.txt",
	})
	if err != nil {
		t.Fatalf("GetArtifactVersion: %v", err)
	}
	av := resp.ArtifactVersion
	if av.Version != 1 {
		t.Fatalf("version = %d, want 1", av.Version)
	}
	if av.CanonicalURI != "s3://b/app/u1/s1/report.txt/1" {
		t.Fatalf("canonicalURI = %q", av.CanonicalURI)
	}
	if av.MimeType != "text/plain" {
		t.Fatalf("mimeType = %q", av.MimeType)
	}
	if av.CustomMetadata["origin"] != "test" {
		t.Fatalf("customMetadata = %v", av.CustomMetadata)
	}
	if av.CreateTime.IsZero() {
		t.Fatal("createTime not populated")
	}
}

func TestSaveInlineBytesKeepsMIMETypeAndSSEKMS(t *testing.T) {
	fake := newFakeS3()
	svc := newTestService(t, fake, Config{Bucket: "b", SSEKMSKeyID: "kms-key-1"})

	if _, err := svc.Save(context.Background(), &artifact.SaveRequest{
		AppName: "app", UserID: "u1", SessionID: "s1", FileName: "img.png",
		Part: genai.NewPartFromBytes([]byte{0x89, 0x50}, "image/png"),
	}); err != nil {
		t.Fatalf("Save: %v", err)
	}

	obj := fake.objects["app/u1/s1/img.png/1"]
	if obj.contentType != "image/png" {
		t.Fatalf("contentType = %q, want image/png", obj.contentType)
	}
	if fake.putSSE != types.ServerSideEncryptionAwsKms || fake.putSSEKeyID != "kms-key-1" {
		t.Fatalf("SSE not applied: %v %q", fake.putSSE, fake.putSSEKeyID)
	}
}

func keys(f *fakeS3) []string {
	var out []string
	for k := range f.objects {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func isFSNotExist(err error) bool {
	return errors.Is(err, fs.ErrNotExist)
}
