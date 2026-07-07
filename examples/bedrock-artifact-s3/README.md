# S3 artifact service

Persists ADK artifacts in Amazon S3 via [`artifact/s3`](../../artifact/s3), instead of the in-memory service the other examples use. The program wires the service the same way you would in `runner.Config{ArtifactService: ...}`, then saves and reloads a text artifact to show the round trip.

Objects are stored one-per-version under `[prefix/]appName/userID/sessionID/fileName/version` (and `.../userID/user/...` for `user:`-scoped filenames), mirroring ADK's upstream GCS artifact service.

## Requirements

- AWS credentials via the default chain
- An S3 bucket you can read/write: `s3:PutObject`, `s3:GetObject`, `s3:DeleteObject`, `s3:ListBucket` on the configured prefix (plus KMS permissions if you set `SSEKMSKeyID`)

## Run

```bash
export ARTIFACT_S3_BUCKET=my-artifacts-bucket
export ARTIFACT_S3_PREFIX=adk-artifacts   # optional
make run
```
