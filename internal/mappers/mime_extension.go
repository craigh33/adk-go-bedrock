package mappers

import (
	"path/filepath"
	"strings"
)

const mimeApplicationOctetStream = "application/octet-stream"

// mimeTypeByExtension maps lowercase extension (with leading dot) to MIME type for Bedrock-oriented uploads.
//
//nolint:gochecknoglobals // Read-only lookup table; a large switch trips gocyclo/cyclop (see git history).
var mimeTypeByExtension = map[string]string{
	// Documents (Bedrock Converse document blocks)
	".pdf":  mimeApplicationPDF,
	".doc":  mimeApplicationDOC,
	".docx": mimeApplicationDOCX,
	".xls":  mimeApplicationXLS,
	".xlsx": mimeApplicationXLSX,
	".csv":  mimeTextCSV,
	".html": mimeTextHTML,
	".htm":  mimeTextHTML,
	".txt":  mimeTextPlain,
	".md":   mimeTextMarkdown,
	".json": mimeApplicationJSON,
	".har":  mimeApplicationJSON,
	".log":  mimeTextPlain,
	".yaml": mimeTextYAML,
	".yml":  mimeTextYAML,
	".xml":  mimeApplicationXML,
	".toml": mimeApplicationTOML,
	".ppt":  mimeApplicationPPT,
	".pptx": mimeApplicationPPTX,
	".jpg":  mimeImageJPEG,
	".jpeg": mimeImageJPEG,
	".png":  mimeImagePNG,
	".gif":  mimeImageGIF,
	".webp": mimeImageWebp,
	".mp3":  mimeAudioMpeg,
	".wav":  mimeAudioWav,
	".ogg":  mimeAudioOgg,
	".flac": mimeAudioFlac,
	".m4a":  mimeAudioM4a,
	".mp4":  mimeVideoMP4,
	".webm": mimeVideoWebm,
	".mov":  mimeVideoQuicktime,
	".mkv":  mimeVideoMatroska,
	// Source, markup, and shell (Bedrock document format txt)
	".go":      mimeTextPlain,
	".rs":      mimeTextPlain,
	".py":      mimeTextPlain,
	".pyw":     mimeTextPlain,
	".pyi":     mimeTextPlain,
	".js":      mimeTextPlain,
	".mjs":     mimeTextPlain,
	".cjs":     mimeTextPlain,
	".jsx":     mimeTextPlain,
	".ts":      mimeTextPlain,
	".tsx":     mimeTextPlain,
	".vue":     mimeTextPlain,
	".svelte":  mimeTextPlain,
	".java":    mimeTextPlain,
	".kt":      mimeTextPlain,
	".kts":     mimeTextPlain,
	".swift":   mimeTextPlain,
	".rb":      mimeTextPlain,
	".rbw":     mimeTextPlain,
	".php":     mimeTextPlain,
	".cs":      mimeTextPlain,
	".fs":      mimeTextPlain,
	".fsx":     mimeTextPlain,
	".fsi":     mimeTextPlain,
	".c":       mimeTextPlain,
	".h":       mimeTextPlain,
	".cc":      mimeTextPlain,
	".cpp":     mimeTextPlain,
	".cxx":     mimeTextPlain,
	".hpp":     mimeTextPlain,
	".hh":      mimeTextPlain,
	".ino":     mimeTextPlain,
	".sh":      mimeTextPlain,
	".bash":    mimeTextPlain,
	".zsh":     mimeTextPlain,
	".fish":    mimeTextPlain,
	".ps1":     mimeTextPlain,
	".psm1":    mimeTextPlain,
	".sql":     mimeTextPlain,
	".graphql": mimeTextPlain,
	".gql":     mimeTextPlain,
	".css":     mimeTextPlain,
	".scss":    mimeTextPlain,
	".less":    mimeTextPlain,
	".r":       mimeTextPlain,
	".lua":     mimeTextPlain,
	".pl":      mimeTextPlain,
	".pm":      mimeTextPlain,
	".vim":     mimeTextPlain,
	".scala":   mimeTextPlain,
	".clj":     mimeTextPlain,
	".cljs":    mimeTextPlain,
	".ex":      mimeTextPlain,
	".exs":     mimeTextPlain,
	".erl":     mimeTextPlain,
	".hrl":     mimeTextPlain,
	".dart":    mimeTextPlain,
}

// MIMETypeFromExtension returns an IANA media type inferred from the path's filename extension
// (case-insensitive). The mapping covers types commonly used with Bedrock Converse (documents,
// images, audio, video). Unrecognized extensions return application/octet-stream.
//
// This is a best-effort hint for clients that only know the file path (e.g. local uploads);
// prefer an explicit MIME from the source (browser File.type, HTTP Content-Type) when available.
func MIMETypeFromExtension(path string) string {
	ext := strings.ToLower(filepath.Ext(path))
	if m, ok := mimeTypeByExtension[ext]; ok {
		return m
	}
	return mimeApplicationOctetStream
}
