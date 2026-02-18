# asr

A suckless, high-performance CLI tool for audio transcription using **Microsoft VibeVoice-ASR**.

It handles local files, direct URLs, and YouTube links with automatic VAD-based chunking for long audio.

## Features

- **engine**: Powered by `microsoft/VibeVoice-ASR`.
- **fast**: Auto-detects **Flash Attention 2** for maximum inference speed on CUDA.
- **smart**: Uses **Silero VAD** to intelligently split long audio (>30m) at silence boundaries, preventing context loss and hallucinations.
- **versatile**:
  - Local files (`.wav`, `.mp3`, `.m4a`, etc.)
  - Direct URLs
  - **YouTube** support (via `yt-dlp`)
- **formats**: Exports to **Text**, **JSON**, or **SRT** (subtitles).
- **diarization**: Speaker-aware output with color-coded terminal display.
- **context**: customizable hotwords/context for better accuracy on proper nouns.

## Installation

Requires **Python 3.12+**, **FFmpeg**, and [uv](https://github.com/astral-sh/uv).

1.  **Clone:**
    ```bash
    git clone https://github.com/federicotorrielli/asr.git
    cd asr
    ```

2.  **Install:**
    ```bash
    uv pip install -e .
    ```

    or just

    ```bash
    uv sync
    ```

    *Note: On CUDA systems, the tool will attempt to auto-install a pre-built `flash-attn` wheel specific to your torch/cuda version on first run.*

## Usage

```bash
asr [INPUT] [OPTIONS]
```

### Examples

**Local file:**
```bash
asr interview.mp3
```

**YouTube video (audio only):**
```bash
asr "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

**Generate subtitles (SRT):**
```bash
asr movie.m4a -f srt -o movie.srt
```

**With context (hotwords):**
```bash
asr meeting.wav -c "Project Gemini, DeepMind, API"
```

**JSON output (machine readable):**
```bash
asr output.wav -f json | jq .
```

### Options

| Option | Description |
| :--- | :--- |
| `INPUT` | Path to file, HTTP URL, or YouTube URL. |
| `-f, --format` | Output format: `text` (default), `json`, `srt`. |
| `-o, --output` | Save output to a specific file. |
| `-c, --context` | Hotwords/context string to guide the model. |
| `--device` | Force device: `auto` (default), `cuda`, `mps`, `cpu`. |
| `--model` | HuggingFace model hub path (default: `microsoft/VibeVoice-ASR`). |
| `--no-timestamps`| Hide timestamps in text output. |
| `--no-speakers` | Hide speaker IDs in text output. |

## Requirements

- `ffmpeg` must be installed and available in your `$PATH` (required for YouTube extraction and audio processing).
