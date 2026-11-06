# Decider payload schema

`openapi.json` is the `/invocations` payload schema exported from
[Strands Decider revision `13c0514`](https://github.com/strands-labs/strands-decider/blob/13c05149265dc404c09d2469636f71b1320e1713/src/strands_decider/schema.py).

Run `make generate-decider` to regenerate the Go models from the pinned snapshot,
or `make check-generated-decider` to check them. Refresh the snapshot only when
updating the supported upstream contract.
