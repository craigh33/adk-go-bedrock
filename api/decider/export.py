# /// script
# requires-python = ">=3.10"
# dependencies = ["fastapi==0.162.0", "pydantic==2.14.1"]
# ///
"""Export only the upstream invocation payload schema; no inference dependencies."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from fastapi import FastAPI

directory = Path(__file__).resolve().parent
source = directory / "upstream/schema.py"
expected = (directory / "upstream/SHA256SUMS").read_text().split()[0]
if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
    raise SystemExit("Upstream schema checksum differs; review the source and provenance before regenerating")
spec = importlib.util.spec_from_file_location("decider_schema", directory / "upstream/schema.py")
schema = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = schema
spec.loader.exec_module(schema)

app = FastAPI(title="Strands Decider invocation payload", version="13c05149265dc404c09d2469636f71b1320e1713")


@app.post("/invocations", response_model=schema.SystemOneResponse)
def invocations(request: schema.SystemOneRequest):
    raise NotImplementedError("Schema export only")


(directory / "openapi.json").write_text(json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n")
