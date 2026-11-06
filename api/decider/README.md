# Decider contract

Payload models come from Strands Decider revision `13c05149265dc404c09d2469636f71b1320e1713`:
[upstream schema](https://github.com/strands-labs/strands-decider/blob/13c05149265dc404c09d2469636f71b1320e1713/src/strands_decider/schema.py).
The unmodified source in `upstream/schema.py` has SHA-256 recorded in `upstream/SHA256SUMS` and is Apache-2.0 licensed (see the repository's LICENSE).

`make generate-decider` requires `uv` and Go. It exports the request/response models through a schema-only FastAPI route and runs pinned oapi-codegen. It does not start a server or load model weights. Python validators are not all represented in OpenAPI; the service retains those boundary checks.

`make check-generated-decider` verifies reproducibility. Builds use committed generated Go code and need no Python or schema access. To update the contract, replace the upstream source from an explicit revision, update its checksum and provenance, regenerate, and review the diff together with validation changes.
