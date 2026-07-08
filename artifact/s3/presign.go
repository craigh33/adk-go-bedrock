package s3artifact

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/s3"
)

const defaultPresignTTL = 15 * time.Minute

// Presigner generates pre-signed S3 GET URLs. Use [PresignClientAdapter] to
// wrap *s3.PresignClient from your configured AWS SDK.
type Presigner interface {
	PresignGetObject(ctx context.Context, bucket, key string, ttl time.Duration) (url string, err error)
}

// PresignClientAdapter adapts *s3.PresignClient to [Presigner].
//
//	pc := s3.NewPresignClient(s3Client)
//	svc, _ := s3artifact.NewService(s3Client, s3artifact.Config{
//	    Bucket:    "my-bucket",
//	    Presigner: s3artifact.PresignClientAdapter{Client: pc},
//	})
type PresignClientAdapter struct {
	Client *s3.PresignClient
}

func (a PresignClientAdapter) PresignGetObject(
	ctx context.Context,
	bucket, key string,
	ttl time.Duration,
) (string, error) {
	if a.Client == nil {
		return "", errors.New("s3artifact: PresignClientAdapter.Client is nil")
	}
	req, err := a.Client.PresignGetObject(ctx, &s3.GetObjectInput{
		Bucket: aws.String(bucket),
		Key:    aws.String(key),
	}, func(opts *s3.PresignOptions) {
		opts.Expires = ttl
	})
	if err != nil {
		return "", err
	}
	return req.URL, nil
}

// PresignLoadRequest is the input to [Service.PresignLoad].
type PresignLoadRequest struct {
	AppName   string
	UserID    string
	SessionID string
	FileName  string
	// Version to sign a URL for. 0 resolves to the latest version.
	Version int64
	// TTL overrides [Config.PresignTTL] for this request.
	// 0 falls back to Config.PresignTTL, then to 15 minutes.
	TTL time.Duration
}

func (r *PresignLoadRequest) validate() error {
	if r.AppName == "" {
		return errors.New("AppName is required")
	}
	if r.UserID == "" {
		return errors.New("UserID is required")
	}
	if r.FileName == "" {
		return errors.New("FileName is required")
	}
	if r.SessionID == "" {
		return errors.New("SessionID is required")
	}
	if r.Version < 0 {
		return errors.New("version must be non-negative")
	}
	if r.TTL < 0 {
		return errors.New("TTL must be non-negative")
	}
	return nil
}

// PresignLoadResponse is the output of [Service.PresignLoad].
type PresignLoadResponse struct {
	// URL is the pre-signed GET URL for the artifact.
	URL string
	// Version is the resolved version the URL points at.
	Version int64
	// Expires is the approximate wall-clock time the URL becomes invalid.
	Expires time.Time
}

// PresignLoad returns a pre-signed GET URL for an artifact. Version 0
// resolves to the latest version. Requires [Config.Presigner] to be set.
func (s *Service) PresignLoad(ctx context.Context, req *PresignLoadRequest) (*PresignLoadResponse, error) {
	if s.cfg.Presigner == nil {
		return nil, errors.New("s3artifact: Presigner not configured in Config")
	}
	if err := req.validate(); err != nil {
		return nil, fmt.Errorf("request validation failed: %w", err)
	}

	version, err := s.resolveVersion(ctx, req.AppName, req.UserID, req.SessionID, req.FileName, req.Version)
	if err != nil {
		return nil, err
	}

	// resolveVersion returns an explicit version as-is without an S3 probe.
	// Verify the object exists so callers get fs.ErrNotExist rather than a
	// signed URL that returns 404 on use.
	if req.Version != 0 {
		key := s.objectKey(req.AppName, req.UserID, req.SessionID, req.FileName, version)
		if _, err := s.client.HeadObject(ctx, &s3.HeadObjectInput{
			Bucket: aws.String(s.cfg.Bucket),
			Key:    aws.String(key),
		}); err != nil {
			if isNotFound(err) {
				return nil, fmt.Errorf("artifact %q version %d not found: %w", req.FileName, version, fs.ErrNotExist)
			}
			return nil, fmt.Errorf("could not verify artifact %q version %d: %w", req.FileName, version, err)
		}
	}

	ttl := req.TTL
	if ttl <= 0 {
		ttl = s.cfg.PresignTTL
	}
	if ttl <= 0 {
		ttl = defaultPresignTTL
	}

	key := s.objectKey(req.AppName, req.UserID, req.SessionID, req.FileName, version)
	url, err := s.cfg.Presigner.PresignGetObject(ctx, s.cfg.Bucket, key, ttl)
	if err != nil {
		return nil, fmt.Errorf("failed to presign artifact URL: %w", err)
	}

	return &PresignLoadResponse{
		URL:     url,
		Version: version,
		Expires: time.Now().Add(ttl),
	}, nil
}
