# asr

A suckless, high-performance CLI tool for audio transcription.

Supports two engines:
- **VibeVoice** (`--engine vibevoice`, default) — Microsoft VibeVoice-ASR with timestamps, speaker diarization, and structured output.
- **Omnilingual** (`--engine omnilingual`) — Meta Omnilingual-ASR supporting 1600+ languages (plain text output).

It handles local files, direct URLs, and YouTube links with automatic VAD-based chunking for long audio.

## Features

- **engines**: VibeVoice-ASR (diarization, timestamps) and Omnilingual-ASR (1600+ languages).
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
    uv sync -U
    ```

    *Note: On CUDA systems, the tool will attempt to auto-install a pre-built `flash-attn` wheel specific to your torch/cuda version on first run.*

    *Note: The omnilingual engine requires `libsndfile` (`apt install libsndfile1` / `brew install libsndfile`).*

## Usage

```bash
uv run asr [INPUT] [OPTIONS]
```

### Examples

**Local file:**
```bash
uv run asr interview.mp3
```

**YouTube video (audio only):**
```bash
uv run asr "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

**Generate subtitles (SRT):**
```bash
uv run asr movie.m4a -f srt -o movie.srt
```

**With context (hotwords):**
```bash
uv run asr meeting.wav -c "Project Gemini, DeepMind, API"
```

**JSON output (machine readable):**
```bash
uv run asr output.wav -f json | jq .
```

**Omnilingual engine (1600+ languages):**
```bash
uv run asr recording.wav -e omnilingual --lang eng_Latn
```

**Omnilingual with Italian:**
```bash
uv run asr audio_italiano.wav -e omnilingual --lang ita_Latn
```

### Options

| Option | Description |
| :--- | :--- |
| `INPUT` | Path to file, HTTP URL, or YouTube URL. |
| `-f, --format` | Output format: `text` (default), `json`, `srt`. |
| `-o, --output` | Save output to a specific file. |
| `-c, --context` | Hotwords/context string to guide the model (vibevoice only). |
| `-e, --engine` | Engine: `vibevoice` (default) or `omnilingual`. |
| `--lang` | Language code for omnilingual (e.g. `eng_Latn`, `ita_Latn`). |
| `--device` | Force device: `auto` (default), `cuda`, `mps`, `cpu`. |
| `--model` | Model name or path (default depends on engine). |
| `--no-timestamps`| Hide timestamps in text output (vibevoice only). |
| `--no-speakers` | Hide speaker IDs in text output (vibevoice only). |

## Requirements

- `ffmpeg` must be installed and available in your `$PATH` (required for YouTube extraction and audio processing).

## Hardware

Running the full model in `bfloat16` requires significant VRAM due to the 7B+ parameter backbone (Qwen2.5-7B) and long-context capabilities.

**VRAM Calculation:**
- **Model Weights**: ~15.6 GB (7.8B params @ 2 bytes/param)
- **CUDA Runtime / Overhead**: ~0.8 GB
- **Activations (Flash Attn)**: ~1.2 GB
- **KV Cache (30 min audio)**: ~1.0 GB (~15k tokens)

**Total Required: ~18.6 GB**

| Component | Recommendation |
| :--- | :--- |
| **VRAM** | **24 GB** (RTX 3090, 4090, or decent server GPU).<br>_16GB cards (4080, 4060 Ti) will **OOM** unless you implement 8-bit/4-bit quantization._ |
| **System RAM** | **32 GB+** (Only relevant if falling back to CPU, which is very slow). |
