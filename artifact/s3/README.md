# s3artifact

Package `s3artifact` provides an Amazon S3 implementation of the ADK
[`artifact.Service`](https://pkg.go.dev/google.golang.org/adk/v2/artifact#Service) interface.

## Storage layout

Each artifact version is a separate S3 object. Keys follow the same convention
as the upstream GCS implementation:

```
[keyPrefix/]<appName>/<userID>/<sessionID>/<fileName>/<version>   # session-scoped
[keyPrefix/]<appName>/<userID>/user/<fileName>/<version>           # user-scoped (fileName has "user:" prefix)
```

S3 bucket versioning is **not** used and is not required.

## Usage

```go
import (
    "github.com/aws/aws-sdk-go-v2/config"
    "github.com/aws/aws-sdk-go-v2/service/s3"
    s3artifact "github.com/craigh33/adk-go-bedrock/artifact/s3"
)

cfg, err := config.LoadDefaultConfig(ctx)
if err != nil { ... }

svc, err := s3artifact.NewService(s3.NewFromConfig(cfg), s3artifact.Config{
    Bucket:      "my-artifact-bucket",
    KeyPrefix:   "adk",          // optional; share a bucket with other data
    SSEKMSKeyID: "arn:aws:kms:…", // optional; leave empty to use bucket default
})
if err != nil { ... }
```

Pass `svc` wherever an `artifact.Service` is required (e.g. `runner.WithArtifactService`).

## IAM permissions

The AWS principal running the service needs the following permissions on the bucket:

| Action | Resource |
|---|---|
| `s3:GetObject`, `s3:PutObject` | `arn:aws:s3:::<bucket>/*` |
| `s3:ListBucket` | `arn:aws:s3:::<bucket>` |
| `s3:DeleteObject` | `arn:aws:s3:::<bucket>/*` |

## Version semantics

- **Auto-increment** (default, `SaveRequest.Version == 0`): next version is `max(existing) + 1`.
- **Explicit version** (`SaveRequest.Version > 0`): the artifact is written at exactly that version. Writing to an already-existing version returns an error wrapping `fs.ErrExist` — versions are immutable once written.
- Concurrent saves of the same artifact with auto-increment can race. Use external coordination if strictly monotonic versions are required.
