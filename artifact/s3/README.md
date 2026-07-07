# s3artifact

`s3artifact` implements `google.golang.org/adk/v2/artifact.Service` backed by Amazon S3.

## Usage

```go
ctx := context.Background()

awsCfg, err := config.LoadDefaultConfig(ctx)
if err != nil {
    log.Fatal(err)
}

svc, err := s3artifact.NewService(
    s3.NewFromConfig(awsCfg),
    s3artifact.Config{
        Bucket:      "my-artifact-bucket",
        KeyPrefix:   "adk",            // optional; share a bucket with other data
        SSEKMSKeyID: "arn:aws:kms:…",  // optional; leave empty to use bucket default encryption
    },
)
if err != nil {
    log.Fatal(err)
}

r, err := runner.New(runner.Config{
    AppName:         "my-app",
    Agent:           agent,
    ArtifactService: svc,
})
```

See [`examples/bedrock-artifact-s3`](../../../../examples/bedrock-artifact-s3) for a runnable setup.

## Required IAM Actions

The AWS principal needs the following permissions on the bucket:

- `s3:GetObject` — `arn:aws:s3:::<bucket>/*`
- `s3:PutObject` — `arn:aws:s3:::<bucket>/*`
- `s3:DeleteObject` — `arn:aws:s3:::<bucket>/*`
- `s3:ListBucket` — `arn:aws:s3:::<bucket>`

## Storage Shape

Artifacts are stored as one S3 object per version. Keys follow the same layout as the upstream GCS implementation:

```
[keyPrefix/]<appName>/<userID>/<sessionID>/<fileName>/<version>   # session-scoped
[keyPrefix/]<appName>/<userID>/user/<fileName>/<version>           # user-scoped (fileName has "user:" prefix)
```

S3 bucket versioning is not used and is not required.

## Version Semantics

- **Auto-increment** (default, `SaveRequest.Version == 0`): next version is `max(existing) + 1`.
- **Explicit version** (`SaveRequest.Version > 0`): written at exactly that version. Writing to an already-existing version returns an error wrapping `fs.ErrExist` — versions are immutable once written.
- Concurrent saves of the same artifact with auto-increment can race. Use external coordination if strictly monotonic versions are required.

## Pre-signed URLs

`PresignLoad` generates a short-lived S3 GET URL so callers can hand a
direct-download link to a client without streaming bytes through the
application layer.

```go
pc := s3.NewPresignClient(s3Client)

svc, err := s3artifact.NewService(s3Client, s3artifact.Config{
    Bucket:     "my-artifact-bucket",
    Presigner:  s3artifact.PresignClientAdapter{Client: pc},
    PresignTTL: 15 * time.Minute, // default; override per-request with PresignLoadRequest.TTL
})

resp, err := svc.PresignLoad(ctx, &s3artifact.PresignLoadRequest{
    AppName: "my-app", UserID: "u1", SessionID: "s1",
    FileName: "report.pdf",   // Version 0 = latest
    TTL:      time.Hour,      // overrides PresignTTL for this request
})
// resp.URL   — signed https://… URL
// resp.Version — resolved version
// resp.Expires — approximate expiry time
```

`Config.Presigner` is opt-in: services that do not set it are unaffected and
`PresignLoad` returns an error if called without one.

No additional IAM actions are needed beyond `s3:GetObject` (already required for
`Load`). The credentials embedded in the signed URL belong to the signer at
signing time.

## Limitations

- Concurrent auto-increment saves of the same artifact are not atomic; two concurrent saves may compute the same next version and one will silently win.
- Listing uses `ListObjectsV2` with a delimiter and requires `s3:ListBucket` on the bucket (not just the prefix).
