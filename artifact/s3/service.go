// Package s3artifact provides an Amazon S3 implementation of ADK's
// [artifact.Service].
//
// Artifacts are stored as one S3 object per version, mirroring ADK's
// gcsartifact layout: an object key of
//
//	[keyPrefix/]appName/userID/sessionID/fileName/version
//
// for session-scoped artifacts, and
//
//	[keyPrefix/]appName/userID/user/fileName/version
//
// for filenames with the "user:" namespace prefix. Versions are ADK's
// explicit int64 versions (a new object per version) — S3 bucket versioning
// is not required and is not consulted.
//
// Like the upstream GCS implementation, Save computes the next version by
// listing existing versions and writing max+1; concurrent saves of the same
// artifact can race. Use external coordination if you need strictly
// monotonic versions under concurrency.
package s3artifact

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"maps"
	"path"
	"slices"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/aws-sdk-go-v2/service/s3/types"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/artifact"
)

// Client is the subset of the S3 API the service uses. *s3.Client satisfies
// it; tests can provide a fake.
type Client interface {
	PutObject(ctx context.Context, params *s3.PutObjectInput, optFns ...func(*s3.Options)) (*s3.PutObjectOutput, error)
	GetObject(ctx context.Context, params *s3.GetObjectInput, optFns ...func(*s3.Options)) (*s3.GetObjectOutput, error)
	HeadObject(ctx context.Context, params *s3.HeadObjectInput, optFns ...func(*s3.Options)) (*s3.HeadObjectOutput, error)
	DeleteObject(ctx context.Context, params *s3.DeleteObjectInput, optFns ...func(*s3.Options)) (*s3.DeleteObjectOutput, error)
	ListObjectsV2(ctx context.Context, params *s3.ListObjectsV2Input, optFns ...func(*s3.Options)) (*s3.ListObjectsV2Output, error)
}

// Config configures the S3 artifact service.
type Config struct {
	// Bucket is the S3 bucket artifacts are stored in. Required.
	Bucket string
	// KeyPrefix is an optional prefix prepended to every object key, letting
	// artifacts share a bucket with other data.
	KeyPrefix string
	// SSEKMSKeyID, when set, enables SSE-KMS with this key on every write.
	// When empty, writes rely on the bucket's default encryption settings.
	SSEKMSKeyID string
}

// Service is an S3-backed [artifact.Service].
type Service struct {
	client Client
	cfg    Config
}

var _ artifact.Service = (*Service)(nil)

// NewService returns an S3-backed [artifact.Service] using the given client
// (typically *s3.Client from your configured AWS SDK).
func NewService(client Client, cfg Config) (*Service, error) {
	if client == nil {
		return nil, errors.New("s3artifact: client is required")
	}
	if cfg.Bucket == "" {
		return nil, errors.New("s3artifact: bucket is required")
	}
	return &Service{client: client, cfg: cfg}, nil
}

// fileHasUserNamespace reports whether a filename is user-scoped rather than
// session-scoped.
func fileHasUserNamespace(fileName string) bool {
	return strings.HasPrefix(fileName, "user:")
}

func (s *Service) objectKey(appName, userID, sessionID, fileName string, version int64) string {
	return s.objectKeyPrefix(appName, userID, sessionID, fileName) + strconv.FormatInt(version, 10)
}

func (s *Service) objectKeyPrefix(appName, userID, sessionID, fileName string) string {
	if fileHasUserNamespace(fileName) {
		return s.withKeyPrefix(fmt.Sprintf("%s/%s/user/%s/", appName, userID, fileName))
	}
	return s.withKeyPrefix(fmt.Sprintf("%s/%s/%s/%s/", appName, userID, sessionID, fileName))
}

func (s *Service) sessionPrefix(appName, userID, sessionID string) string {
	return s.withKeyPrefix(fmt.Sprintf("%s/%s/%s/", appName, userID, sessionID))
}

func (s *Service) userPrefix(appName, userID string) string {
	return s.withKeyPrefix(fmt.Sprintf("%s/%s/user/", appName, userID))
}

func (s *Service) withKeyPrefix(key string) string {
	if s.cfg.KeyPrefix == "" {
		return key
	}
	return strings.TrimSuffix(s.cfg.KeyPrefix, "/") + "/" + key
}

// Save implements [artifact.Service]. Note the version race documented on the
// package: the next version is max(existing)+1 from a list call.
func (s *Service) Save(ctx context.Context, req *artifact.SaveRequest) (*artifact.SaveResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, fmt.Errorf("request validation failed: %w", err)
	}

	nextVersion := int64(1)
	existing, err := s.versions(ctx, &artifact.VersionsRequest{
		AppName: req.AppName, UserID: req.UserID, SessionID: req.SessionID, FileName: req.FileName,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to list artifact versions: %w", err)
	}
	if len(existing.Versions) > 0 {
		nextVersion = slices.Max(existing.Versions) + 1
	}

	data, contentType := partPayload(req.Part)
	put := &s3.PutObjectInput{
		Bucket:      aws.String(s.cfg.Bucket),
		Key:         aws.String(s.objectKey(req.AppName, req.UserID, req.SessionID, req.FileName, nextVersion)),
		Body:        bytes.NewReader(data),
		ContentType: aws.String(contentType),
	}
	if s.cfg.SSEKMSKeyID != "" {
		put.ServerSideEncryption = types.ServerSideEncryptionAwsKms
		put.SSEKMSKeyId = aws.String(s.cfg.SSEKMSKeyID)
	}
	if _, err := s.client.PutObject(ctx, put); err != nil {
		return nil, fmt.Errorf("failed to write artifact to S3: %w", err)
	}
	return &artifact.SaveResponse{Version: nextVersion}, nil
}

// partPayload extracts bytes + MIME type from a genai.Part the way ADK's
// artifact services expect: inline bytes with their MIME type, or text as
// text/plain.
func partPayload(part *genai.Part) ([]byte, string) {
	if part.InlineData != nil {
		return part.InlineData.Data, part.InlineData.MIMEType
	}
	return []byte(part.Text), "text/plain"
}

// Load implements [artifact.Service]. Version 0 loads the latest version.
func (s *Service) Load(ctx context.Context, req *artifact.LoadRequest) (_ *artifact.LoadResponse, err error) {
	if err := req.Validate(); err != nil {
		return nil, fmt.Errorf("request validation failed: %w", err)
	}
	version, err := s.resolveVersion(ctx, req.AppName, req.UserID, req.SessionID, req.FileName, req.Version)
	if err != nil {
		return nil, err
	}

	key := s.objectKey(req.AppName, req.UserID, req.SessionID, req.FileName, version)
	out, err := s.client.GetObject(ctx, &s3.GetObjectInput{
		Bucket: aws.String(s.cfg.Bucket),
		Key:    aws.String(key),
	})
	if err != nil {
		if isNotFound(err) {
			return nil, fmt.Errorf("artifact %q not found: %w", key, fs.ErrNotExist)
		}
		return nil, fmt.Errorf("could not get artifact object %q: %w", key, err)
	}
	defer func() {
		if closeErr := out.Body.Close(); closeErr != nil && err == nil {
			err = fmt.Errorf("failed to close object body: %w", closeErr)
		}
	}()

	data, err := io.ReadAll(out.Body)
	if err != nil {
		return nil, fmt.Errorf("could not read artifact object %q: %w", key, err)
	}
	return &artifact.LoadResponse{Part: genai.NewPartFromBytes(data, aws.ToString(out.ContentType))}, nil
}

// Delete implements [artifact.Service]. Version 0 deletes every version;
// deleting a specific version exposes the previous one as latest.
func (s *Service) Delete(ctx context.Context, req *artifact.DeleteRequest) error {
	if err := req.Validate(); err != nil {
		return fmt.Errorf("request validation failed: %w", err)
	}

	deleteVersion := func(v int64) error {
		key := s.objectKey(req.AppName, req.UserID, req.SessionID, req.FileName, v)
		if _, err := s.client.DeleteObject(ctx, &s3.DeleteObjectInput{
			Bucket: aws.String(s.cfg.Bucket),
			Key:    aws.String(key),
		}); err != nil {
			return fmt.Errorf("failed to delete artifact %q: %w", key, err)
		}
		return nil
	}

	if req.Version != 0 {
		return deleteVersion(req.Version)
	}

	existing, err := s.versions(ctx, &artifact.VersionsRequest{
		AppName: req.AppName, UserID: req.UserID, SessionID: req.SessionID, FileName: req.FileName,
	})
	if err != nil {
		return fmt.Errorf("failed to fetch versions on delete artifact: %w", err)
	}
	for _, v := range existing.Versions {
		if err := deleteVersion(v); err != nil {
			return err
		}
	}
	return nil
}

// List implements [artifact.Service]: the union of session-scoped and
// user-scoped filenames, sorted.
func (s *Service) List(ctx context.Context, req *artifact.ListRequest) (*artifact.ListResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, fmt.Errorf("request validation failed: %w", err)
	}

	filenames := map[string]bool{}
	for _, prefix := range []string{
		s.sessionPrefix(req.AppName, req.UserID, req.SessionID),
		s.userPrefix(req.AppName, req.UserID),
	} {
		if err := s.eachKey(ctx, prefix, func(key string) {
			// Key shape: .../fileName/version — the filename is the
			// second-to-last segment (path separators in filenames are
			// rejected at validation).
			segments := strings.Split(key, "/")
			if len(segments) >= 2 {
				filenames[segments[len(segments)-2]] = true
			}
		}); err != nil {
			return nil, fmt.Errorf("failed to list artifacts under %q: %w", prefix, err)
		}
	}

	names := slices.Collect(maps.Keys(filenames))
	sort.Strings(names)
	return &artifact.ListResponse{FileNames: names}, nil
}

// Versions implements [artifact.Service] and errors when the artifact has no
// versions.
func (s *Service) Versions(ctx context.Context, req *artifact.VersionsRequest) (*artifact.VersionsResponse, error) {
	resp, err := s.versions(ctx, req)
	if err != nil {
		return nil, err
	}
	if len(resp.Versions) == 0 {
		return nil, fmt.Errorf("artifact not found: %w", fs.ErrNotExist)
	}
	return resp, nil
}

// versions lists an artifact's versions, returning an empty slice (not an
// error) when none exist.
func (s *Service) versions(ctx context.Context, req *artifact.VersionsRequest) (*artifact.VersionsResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, fmt.Errorf("request validation failed: %w", err)
	}

	versions := make([]int64, 0)
	prefix := s.objectKeyPrefix(req.AppName, req.UserID, req.SessionID, req.FileName)
	if err := s.eachKey(ctx, prefix, func(key string) {
		v, err := strconv.ParseInt(path.Base(key), 10, 64)
		if err != nil {
			return // non-numeric trailing segment: not one of our version objects
		}
		versions = append(versions, v)
	}); err != nil {
		return nil, fmt.Errorf("failed to list artifact versions under %q: %w", prefix, err)
	}
	return &artifact.VersionsResponse{Versions: versions}, nil
}

// GetArtifactVersion implements [artifact.Service]. Version 0 resolves to the
// latest version.
func (s *Service) GetArtifactVersion(ctx context.Context, req *artifact.GetArtifactVersionRequest) (*artifact.GetArtifactVersionResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, fmt.Errorf("request validation failed: %w", err)
	}
	version, err := s.resolveVersion(ctx, req.AppName, req.UserID, req.SessionID, req.FileName, req.Version)
	if err != nil {
		return nil, err
	}

	key := s.objectKey(req.AppName, req.UserID, req.SessionID, req.FileName, version)
	head, err := s.client.HeadObject(ctx, &s3.HeadObjectInput{
		Bucket: aws.String(s.cfg.Bucket),
		Key:    aws.String(key),
	})
	if err != nil {
		if isNotFound(err) {
			return nil, fmt.Errorf("artifact %q not found: %w", key, fs.ErrNotExist)
		}
		return nil, fmt.Errorf("could not head artifact object %q: %w", key, err)
	}

	customMeta := make(map[string]any, len(head.Metadata))
	for k, v := range head.Metadata {
		customMeta[k] = v
	}

	var createTime time.Time
	if head.LastModified != nil {
		createTime = *head.LastModified
	}

	return &artifact.GetArtifactVersionResponse{
		ArtifactVersion: &artifact.ArtifactVersion{
			Version:        version,
			CanonicalURI:   fmt.Sprintf("s3://%s/%s", s.cfg.Bucket, key),
			CustomMetadata: customMeta,
			CreateTime:     createTime,
			MimeType:       aws.ToString(head.ContentType),
		},
	}, nil
}

// resolveVersion returns the given version, or the latest when it is 0.
func (s *Service) resolveVersion(ctx context.Context, appName, userID, sessionID, fileName string, version int64) (int64, error) {
	if version != 0 {
		return version, nil
	}
	resp, err := s.versions(ctx, &artifact.VersionsRequest{
		AppName: appName, UserID: userID, SessionID: sessionID, FileName: fileName,
	})
	if err != nil {
		return 0, fmt.Errorf("failed to list artifact versions: %w", err)
	}
	if len(resp.Versions) == 0 {
		return 0, fmt.Errorf("artifact not found: %w", fs.ErrNotExist)
	}
	return slices.Max(resp.Versions), nil
}

// eachKey calls fn with every object key under prefix, following list
// pagination.
func (s *Service) eachKey(ctx context.Context, prefix string, fn func(key string)) error {
	var continuation *string
	for {
		out, err := s.client.ListObjectsV2(ctx, &s3.ListObjectsV2Input{
			Bucket:            aws.String(s.cfg.Bucket),
			Prefix:            aws.String(prefix),
			ContinuationToken: continuation,
		})
		if err != nil {
			return err
		}
		for _, obj := range out.Contents {
			fn(aws.ToString(obj.Key))
		}
		if !aws.ToBool(out.IsTruncated) {
			return nil
		}
		continuation = out.NextContinuationToken
	}
}

// isNotFound reports whether an S3 error means the object does not exist.
func isNotFound(err error) bool {
	var noKey *types.NoSuchKey
	var notFound *types.NotFound
	return errors.As(err, &noKey) || errors.As(err, &notFound)
}
