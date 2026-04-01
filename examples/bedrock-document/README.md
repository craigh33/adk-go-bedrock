# bedrock-document — document upload debugging

Small CLI to exercise **inline document** mapping (`genai.Part` → Bedrock Converse) and optional model calls. Use this when debugging issues like Web UI uploads (PDF/DOCX) or verifying MIME/filename handling **without** running the full ADK server.

MIME types for `-path` are inferred with [`mappers.MIMETypeFromExtension`](../../bedrock/mappers/mime_extension.go) (same helper the mapper uses when resolving `application/octet-stream` from a filename).

## Requirements

Same as other examples: AWS credentials, region, and a **multimodal** Bedrock model that accepts documents for your account (see AWS docs for document support per model).

**Not supported as documents:** PowerPoint (`.ppt` / `.pptx`). Bedrock Converse document formats are PDF, Word, Excel, CSV, HTML, plain text, and Markdown only—try exporting slides to PDF or copying content into a supported format. The mapper returns a clear error if you pass `.ppt`/`.pptx` (including `application/octet-stream` with a `.pptx` filename).

## Commands

**Map-only (no AWS credentials or `InvokeModel` — validates `bedrock/mappers` only):**

```bash
go run . -dry-run -path /path/to/file.docx
```

**Unary Converse:**

```bash
export BEDROCK_MODEL_ID=eu.amazon.nova-2-lite-v1:0
export AWS_REGION=eu-west-1
go run . -path ./sample.pdf
```

**Streaming:**

```bash
go run . -stream -path ./sample.pdf
```

**Same `Part` for text + file** (matches some clients that bundle prompt and attachment):

```bash
go run . -dry-run -combined -path ./report.docx
go run . -combined -path ./report.docx
```

**Custom prompt:**

```bash
go run . -path ./memo.pdf -prompt "List three bullet points from this document."
```

Environment variable **`DOCUMENT_PATH`** is used if `-path` is omitted.

## What to look for

- **`-dry-run`** prints Bedrock message count and block types (`document`, `text`, etc.). If mapping fails, you get an error immediately (no invocation ID on the server — same class of failure as a bad request body).
- **MIME** is inferred from the file extension in this example (see the `mappers.MIMETypeFromExtension` call in `main.go`). Your browser may send different types; the mapper also normalizes parameters and infers from `application/zip` / `application/octet-stream` when the filename is present.
- On failure, the process logs **`GenerateContent: ...`** or **`mapping failed: ...`** with the underlying error (validation, AWS API, etc.).

## Makefile

```bash
make -C examples/bedrock-document run ARGS='-dry-run -path ./file.docx'
```
