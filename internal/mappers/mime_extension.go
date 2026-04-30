package mappers

import (
	"path/filepath"
	"strings"
)

const mimeApplicationOctetStream = "application/octet-stream"

// MIMETypeFromExtension returns an IANA media type inferred from the path's filename extension
// (case-insensitive). The mapping covers types commonly used with Bedrock Converse (documents,
// images, audio, video). Unrecognized extensions return application/octet-stream.
//
// This is a best-effort hint for clients that only know the file path (e.g. local uploads);
// prefer an explicit MIME from the source (browser File.type, HTTP Content-Type) when available.
//
//nolint:funlen // Large inline extension→MIME table to avoid a package-level map that would trip gochecknoglobals.
func MIMETypeFromExtension(path string) string {
	switch strings.ToLower(filepath.Ext(path)) {
	// Documents (Bedrock Converse document blocks)
	case ".pdf":
		return mimeApplicationPDF
	case ".doc":
		return mimeApplicationDOC
	case ".docx":
		return mimeApplicationDOCX
	case ".xls":
		return mimeApplicationXLS
	case ".xlsx":
		return mimeApplicationXLSX
	case ".csv":
		return mimeTextCSV
	case ".html", ".htm":
		return mimeTextHTML
	case ".txt":
		return mimeTextPlain
	case ".md":
		return mimeTextMarkdown
	case ".json", ".har":
		return mimeApplicationJSON
	case ".log":
		return mimeTextPlain
	case ".yaml", ".yml":
		return mimeTextYAML
	case ".xml":
		return mimeApplicationXML
	case ".toml":
		return mimeApplicationTOML
	// Source, markup, and shell (Bedrock document format txt)
	case ".go", ".rs", ".py", ".pyw", ".pyi", ".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx", ".vue", ".svelte",
		".java", ".kt", ".kts", ".swift", ".rb", ".rbw", ".php", ".cs", ".fs", ".fsx", ".fsi",
		".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".ino",
		".sh", ".bash", ".zsh", ".fish", ".ps1", ".psm1",
		".sql", ".graphql", ".gql",
		".css", ".scss", ".less",
		".r", ".lua", ".pl", ".pm", ".vim",
		".scala", ".clj", ".cljs", ".ex", ".exs", ".erl", ".hrl", ".dart":
		return mimeTextPlain
	case ".ppt":
		return mimeApplicationPPT
	case ".pptx":
		return mimeApplicationPPTX
	// Images (Bedrock image blocks)
	case ".jpg", ".jpeg":
		return mimeImageJPEG
	case ".png":
		return mimeImagePNG
	case ".gif":
		return mimeImageGIF
	case ".webp":
		return mimeImageWebp
	// Audio (Bedrock audio blocks)
	case ".mp3":
		return mimeAudioMpeg
	case ".wav":
		return mimeAudioWav
	case ".ogg":
		return mimeAudioOgg
	case ".flac":
		return mimeAudioFlac
	case ".m4a":
		return mimeAudioM4a
	// Video (Bedrock video blocks)
	case ".mp4":
		return mimeVideoMP4
	case ".webm":
		return mimeVideoWebm
	case ".mov":
		return mimeVideoQuicktime
	case ".mkv":
		return mimeVideoMatroska
	default:
		return mimeApplicationOctetStream
	}
}
