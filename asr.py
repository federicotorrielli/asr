"""ASR CLI — Transcribe audio from files or URLs.

Supported engines:
  - vibevoice: Microsoft VibeVoice-ASR (timestamps, speakers, diarization)
  - omnilingual: Meta Omnilingual-ASR (1600+ languages, plain text)
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import click
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

SPEAKER_COLORS = [
    "cyan",
    "green",
    "yellow",
    "magenta",
    "blue",
    "red",
    "bright_cyan",
    "bright_green",
    "bright_yellow",
    "bright_magenta",
]

MODEL_DEFAULT = "microsoft/VibeVoice-ASR"
OMNI_MODEL_DEFAULT = "omniASR_LLM_Unlimited_7B_v2"


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def detect_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.bfloat16
    return torch.float32


def detect_attention(device: str) -> str:
    if "cuda" in device:
        try:
            import flash_attn  # noqa: F401

            return "flash_attention_2"
        except ImportError:
            # Try to auto-install
            try:
                from flash_attn_setup import ensure_flash_attn

                if ensure_flash_attn():
                    import importlib

                    importlib.invalidate_caches()
                    try:
                        import flash_attn  # noqa: F401

                        return "flash_attention_2"
                    except ImportError:
                        console.print(
                            "[yellow]⚠  Auto-installed flash-attn but import failed. Using SDPA.[/]"
                        )
            except Exception as e:
                console.print(
                    f"[yellow]⚠  Could not auto-install flash-attn ({e}). Using SDPA.[/]"
                )

    return "sdpa"


def is_url(s: str) -> bool:
    parsed = urlparse(s)
    return parsed.scheme in ("http", "https")


def is_youtube(url: str) -> bool:
    host = urlparse(url).hostname or ""
    return any(h in host for h in ("youtube.com", "youtu.be", "youtube-nocookie.com"))


def download_youtube(url: str, tmp_dir: str) -> Path:
    """Download audio from YouTube using yt-dlp."""
    import yt_dlp

    out_path = os.path.join(tmp_dir, "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "audio")

    # yt-dlp saves as .wav after postprocessing
    for f in Path(tmp_dir).iterdir():
        if f.suffix == ".wav":
            return f

    raise FileNotFoundError(f"Failed to extract audio from: {url} (title: {title})")


def download_url(url: str, tmp_dir: str) -> Path:
    """Download audio from a direct URL."""
    import httpx

    out_path = Path(tmp_dir) / "download_audio"
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        ct = r.headers.get("content-type", "")
        ext = ".wav"
        if "mp3" in ct or url.endswith(".mp3"):
            ext = ".mp3"
        elif "flac" in ct or url.endswith(".flac"):
            ext = ".flac"
        elif "mp4" in ct or url.endswith(".m4a"):
            ext = ".m4a"
        elif "ogg" in ct or url.endswith(".ogg"):
            ext = ".ogg"
        elif "webm" in ct or url.endswith(".webm"):
            ext = ".webm"
        out_path = out_path.with_suffix(ext)
        with open(out_path, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=65536):
                f.write(chunk)
    return out_path


def resolve_input(source: str, tmp_dir: str) -> Path:
    """Resolve input source to a local file path."""
    if is_url(source):
        if is_youtube(source):
            with console.status(
                "[bold cyan]⬇  Downloading from YouTube…", spinner="dots"
            ):
                return download_youtube(source, tmp_dir)
        else:
            with console.status("[bold cyan]⬇  Downloading audio…", spinner="dots"):
                return download_url(source, tmp_dir)

    path = Path(source).expanduser().resolve()
    if not path.exists():
        console.print(f"[bold red]✗[/] File not found: {path}")
        sys.exit(1)
    return path


# Formats natively supported by libsndfile (soundfile)
_SOUNDFILE_EXTS = {
    ".wav",
    ".flac",
    ".ogg",
    ".oga",
    ".aiff",
    ".aif",
    ".aifc",
    ".raw",
    ".rf64",
    ".w64",
    ".caf",
    ".sd2",
    ".au",
    ".snd",
    ".nist",
    ".voc",
    ".pvf",
    ".xi",
    ".htk",
    ".sds",
    ".avr",
    ".wavex",
    ".svx",
    ".mpc2k",
    ".mat4",
    ".mat5",
    ".ircam",
}


def ensure_supported_format(audio_path: Path, tmp_dir: str) -> Path:
    """Convert audio to WAV via ffmpeg if the format isn't supported by soundfile."""
    if audio_path.suffix.lower() in _SOUNDFILE_EXTS:
        return audio_path

    import subprocess

    wav_path = Path(tmp_dir) / (audio_path.stem + ".wav")
    with console.status(
        f"[bold cyan]🔄 Converting {audio_path.suffix} → .wav…", spinner="dots"
    ):
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(audio_path),
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_s16le",
                    str(wav_path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
        except FileNotFoundError:
            console.print(
                "[bold red]✗[/] ffmpeg not found. "
                "Install it to handle m4a/mp3/webm files: "
                "[bold]apt install ffmpeg[/] or [bold]brew install ffmpeg[/]"
            )
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            console.print(f"[bold red]✗[/] ffmpeg conversion failed:\n{e.stderr}")
            sys.exit(1)

    console.print("[green]✓[/] Converted to WAV")
    return wav_path


MAX_CHUNK_S = 30 * 60  # 30 minutes — model handles 60, so 30 gives headroom
MIN_SILENCE_MS = 700  # minimum silence gap to consider as a split candidate
VAD_SAMPLE_RATE = 16000  # silero-vad native sample rate


def get_audio_duration(audio_path: Path) -> float:
    """Get audio duration in seconds using soundfile."""
    import soundfile as sf

    info = sf.info(str(audio_path))
    return info.duration


def _load_vad_model() -> Any:
    """Load silero-vad model (cached by torch.hub)."""
    from silero_vad import load_silero_vad

    return load_silero_vad()


def find_split_points(audio_path: Path) -> list[float]:
    """Use silero-vad to find optimal split points at silence boundaries.

    Returns a sorted list of timestamps (seconds) where the audio should be split.
    Greedily accumulates speech segments into chunks ≤ MAX_CHUNK_S, splitting at
    the largest silence gap when a chunk would exceed the limit.
    """
    import torchaudio

    waveform, sr = torchaudio.load(str(audio_path))
    # silero-vad expects mono 16kHz
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != VAD_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, VAD_SAMPLE_RATE)
    waveform = waveform.squeeze(0)

    vad_model = _load_vad_model()

    from silero_vad import get_speech_timestamps

    speech_ts = get_speech_timestamps(
        waveform,
        vad_model,
        sampling_rate=VAD_SAMPLE_RATE,
        min_silence_duration_ms=MIN_SILENCE_MS,
        return_seconds=True,
    )

    if not speech_ts:
        return []

    # Build silence gaps between consecutive speech segments.
    # Each gap: (start_s, end_s, duration_s, position_in_sequence)
    gaps: list[tuple[float, float, float]] = []
    for i in range(len(speech_ts) - 1):
        gap_start = speech_ts[i]["end"]
        gap_end = speech_ts[i + 1]["start"]
        gaps.append((gap_start, gap_end, gap_end - gap_start))

    total_duration = waveform.shape[0] / VAD_SAMPLE_RATE
    split_points: list[float] = []
    chunk_start = 0.0

    for i, gap in enumerate(gaps):
        gap_mid = (gap[0] + gap[1]) / 2
        chunk_end_if_continued = speech_ts[i + 1]["end"]

        if chunk_end_if_continued - chunk_start > MAX_CHUNK_S:
            # This chunk would exceed max — split at this gap's midpoint
            split_points.append(gap_mid)
            chunk_start = gap_mid

    # Handle edge case: last segment extends past MAX_CHUNK_S from last split
    last_start = split_points[-1] if split_points else 0.0
    if total_duration - last_start > MAX_CHUNK_S:
        # No good silence gap found in the tail — force split at MAX_CHUNK_S intervals
        pos = last_start + MAX_CHUNK_S
        while pos < total_duration:
            split_points.append(pos)
            pos += MAX_CHUNK_S

    return sorted(split_points)


def chunk_audio_vad(audio_path: Path, chunk_dir: str) -> list[tuple[Path, float]]:
    """Split audio at silence boundaries using VAD. Returns (path, offset_seconds) pairs.

    Falls back to fixed-size splitting if VAD fails.
    """
    import soundfile as sf

    info = sf.info(str(audio_path))
    sr = info.samplerate
    total_frames = info.frames
    duration = info.duration

    try:
        split_points = find_split_points(audio_path)
    except Exception as e:
        console.print(f"[yellow]⚠  VAD failed ({e}), falling back to fixed splits[/]")
        split_points = []
        pos = float(MAX_CHUNK_S)
        while pos < duration:
            split_points.append(pos)
            pos += MAX_CHUNK_S

    if not split_points:
        # Audio fits in a single chunk
        return [(audio_path, 0.0)]

    # Build boundary list: [0, split1, split2, ..., duration]
    boundaries = [0.0] + split_points + [duration]
    chunks: list[tuple[Path, float]] = []

    with sf.SoundFile(str(audio_path), "r") as src:
        for i in range(len(boundaries) - 1):
            start_s = boundaries[i]
            end_s = boundaries[i + 1]
            start_frame = int(start_s * sr)
            end_frame = min(int(end_s * sr), total_frames)
            n_frames = end_frame - start_frame

            src.seek(start_frame)
            data = src.read(n_frames)

            chunk_path = Path(chunk_dir) / f"chunk_{i:03d}.wav"
            sf.write(str(chunk_path), data, sr)
            chunks.append((chunk_path, start_s))

    return chunks


def offset_timestamp(ts: str | int | float, offset_s: float) -> str:
    """Add offset_s seconds to a timestamp string like 'MM:SS.ss' or 'HH:MM:SS.ss'."""
    try:
        if isinstance(ts, (int, float)):
            total = float(ts) + offset_s
        else:
            parts = ts.replace(",", ".").split(":")
            if len(parts) == 2:
                total = int(parts[0]) * 60 + float(parts[1])
            elif len(parts) == 3:
                total = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            else:
                return ts

            total += offset_s
        h = int(total // 3600)
        m = int((total % 3600) // 60)
        s = total % 60
        if h > 0:
            return f"{h}:{m:02d}:{s:05.2f}"
        return f"{m}:{s:05.2f}"
    except (ValueError, TypeError):
        return ts


def transcribe_chunked(
    model: Any,
    processor: Any,
    audio_path: Path,
    device: str,
    max_new_tokens: int,
    context: str | None,
    tmp_dir: str,
) -> dict[str, Any]:
    """Transcribe audio, intelligently splitting into chunks at silence boundaries."""
    duration = get_audio_duration(audio_path)

    if duration <= MAX_CHUNK_S:
        return transcribe(model, processor, audio_path, device, max_new_tokens, context)

    console.print(
        f"[yellow]⚡[/] Audio is {duration / 60:.0f}min — "
        f"analyzing for intelligent split points…"
    )

    chunk_dir = os.path.join(tmp_dir, "chunks")
    os.makedirs(chunk_dir, exist_ok=True)

    with console.status(
        "[bold cyan]🔍 Running VAD to find silence boundaries…", spinner="dots"
    ):
        chunks = chunk_audio_vad(audio_path, chunk_dir)

    console.print(f"[green]✓[/] Split into {len(chunks)} chunks at silence boundaries")

    all_segments: list[dict] = []
    all_raw: list[str] = []
    total_elapsed = 0.0

    for i, (chunk_path, offset) in enumerate(chunks):
        chunk_dur = get_audio_duration(chunk_path)
        console.print(
            f"\n[bold cyan]🎙  Chunk {i + 1}/{len(chunks)}[/] "
            f"[dim]({offset / 60:.1f}–{(offset + chunk_dur) / 60:.1f}min)[/]"
        )
        with console.status("[bold cyan]  Transcribing…", spinner="dots"):
            result = transcribe(
                model, processor, chunk_path, device, max_new_tokens, context
            )

        total_elapsed += result["elapsed"]
        all_raw.append(result["raw_text"])

        for seg in result["segments"]:
            if "start_time" in seg:
                seg["start_time"] = offset_timestamp(seg["start_time"], offset)
            if "end_time" in seg:
                seg["end_time"] = offset_timestamp(seg["end_time"], offset)
            all_segments.append(seg)

        console.print(
            f"  [green]✓[/] {len(result['segments'])} segments "
            f"[dim]({result['elapsed']:.1f}s)[/]"
        )

    return {
        "raw_text": "\n".join(all_raw),
        "segments": all_segments,
        "elapsed": total_elapsed,
    }


def load_model(
    model_path: str, device: str, dtype: torch.dtype, attn: str
) -> tuple[Any, Any]:
    """Load VibeVoice-ASR model and processor."""
    from vibevoice.modular.modeling_vibevoice_asr import (
        VibeVoiceASRForConditionalGeneration,
    )
    from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

    processor = VibeVoiceASRProcessor.from_pretrained(
        model_path, language_model_pretrained_name="Qwen/Qwen2.5-7B"
    )

    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=device if device == "auto" else None,
        attn_implementation=attn,
        trust_remote_code=True,
    )
    if device != "auto":
        model = model.to(device)

    model.eval()
    return model, processor


def transcribe(
    model: Any,
    processor: Any,
    audio_path: Path,
    device: str,
    max_new_tokens: int,
    context: str | None,
) -> dict[str, Any]:
    """Run transcription on a single audio file."""
    inputs = processor(
        audio=str(audio_path),
        sampling_rate=None,
        return_tensors="pt",
        padding=True,
        add_generation_prompt=True,
        context_info=context,
    )
    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
    }

    gen_config = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": processor.pad_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
        "do_sample": False,
    }

    t0 = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_config)
    elapsed = time.perf_counter() - t0

    input_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, input_length:]

    eos_positions = (generated_ids == processor.tokenizer.eos_token_id).nonzero(
        as_tuple=True
    )[0]
    if len(eos_positions) > 0:
        generated_ids = generated_ids[: eos_positions[0] + 1]

    raw_text = processor.decode(generated_ids, skip_special_tokens=True)

    try:
        segments = processor.post_process_transcription(raw_text)
    except Exception:
        segments = []

    return {
        "raw_text": raw_text,
        "segments": segments,
        "elapsed": elapsed,
    }


def load_model_omni(model_card: str) -> Any:
    """Load an Omnilingual-ASR inference pipeline."""
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    return ASRInferencePipeline(model_card=model_card)


def transcribe_omni(
    pipeline: Any,
    audio_path: Path,
    lang: str | None,
) -> dict[str, Any]:
    """Run transcription using the Omnilingual-ASR pipeline."""
    kwargs: dict[str, Any] = {"batch_size": 1}
    if lang:
        kwargs["lang"] = [lang]

    t0 = time.perf_counter()
    results = pipeline.transcribe([str(audio_path)], **kwargs)
    elapsed = time.perf_counter() - t0

    raw_text = results[0] if results else ""
    return {
        "raw_text": raw_text,
        "segments": [],
        "elapsed": elapsed,
    }


def format_timestamp_srt(ts: str) -> str:
    """Convert 'MM:SS.ss' or 'HH:MM:SS.ss' to SRT format 'HH:MM:SS,mmm'."""
    try:
        parts = ts.replace(",", ".").split(":")
        if len(parts) == 2:
            m, s = parts
            h = 0
        elif len(parts) == 3:
            h, m, s = parts
        else:
            return ts

        h, m = int(h), int(m)
        s_float = float(s)
        s_int = int(s_float)
        ms = int((s_float - s_int) * 1000)
        return f"{h:02d}:{m:02d}:{s_int:02d},{ms:03d}"
    except (ValueError, TypeError):
        return ts


def render_text(
    segments: list[dict],
    raw_text: str,
    show_timestamps: bool,
    show_speakers: bool,
) -> str:
    """Render transcription as human-readable text."""
    if not segments:
        return raw_text

    lines = []
    for seg in segments:
        parts = []
        if show_timestamps:
            start = seg.get("start_time", "")
            end = seg.get("end_time", "")
            if start or end:
                parts.append(f"[{start} → {end}]")
        if show_speakers:
            speaker = seg.get("speaker_id", "")
            if speaker:
                parts.append(f"Speaker {speaker}:")
        parts.append(seg.get("text", ""))
        lines.append(" ".join(parts))

    return "\n".join(lines)


def render_srt(segments: list[dict]) -> str:
    """Render transcription as SRT subtitles."""
    blocks = []
    for i, seg in enumerate(segments, 1):
        start = format_timestamp_srt(seg.get("start_time", "00:00.00"))
        end = format_timestamp_srt(seg.get("end_time", "00:00.00"))
        text = seg.get("text", "")
        speaker = seg.get("speaker_id", "")
        content = f"[Speaker {speaker}] {text}" if speaker else text
        blocks.append(f"{i}\n{start} --> {end}\n{content}")
    return "\n\n".join(blocks)


def display_rich(segments: list[dict], raw_text: str, elapsed: float) -> None:
    """Display transcription with Rich formatting."""
    console.print()

    if not segments:
        console.print(Panel(raw_text, title="Transcription", border_style="green"))
        console.print(f"\n[dim]⏱  Completed in {elapsed:.1f}s[/]")
        return

    table = Table(
        show_header=True,
        header_style="bold bright_white",
        border_style="bright_black",
        title="[bold]Transcription[/]",
        title_style="bold bright_cyan",
        padding=(0, 1),
        expand=True,
    )
    table.add_column("Time", style="dim", width=20, no_wrap=True)
    table.add_column("Speaker", width=10, no_wrap=True)
    table.add_column("Content", ratio=1)

    speaker_color_map: dict[str, str] = {}

    for seg in segments:
        start = seg.get("start_time", "")
        end = seg.get("end_time", "")
        speaker = str(seg.get("speaker_id", ""))
        text = seg.get("text", "")

        if speaker and speaker not in speaker_color_map:
            color_idx = len(speaker_color_map) % len(SPEAKER_COLORS)
            speaker_color_map[speaker] = SPEAKER_COLORS[color_idx]

        color = speaker_color_map.get(speaker, "white")
        time_str = f"{start} → {end}" if start or end else ""

        table.add_row(
            time_str,
            Text(f"Speaker {speaker}" if speaker else "", style=f"bold {color}"),
            Text(text, style=color),
        )

    console.print(table)
    console.print(f"\n[dim]📊 {len(segments)} segments  •  ⏱  {elapsed:.1f}s[/]")


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.argument("input_source")
@click.option(
    "-c",
    "--context",
    default=None,
    help="Hotwords or context to improve accuracy (e.g. 'John, AI, Microsoft').",
)
@click.option(
    "-f",
    "--format",
    "fmt",
    type=click.Choice(["text", "json", "srt"], case_sensitive=False),
    default="text",
    show_default=True,
    help="Output format.",
)
@click.option(
    "-o", "--output", "output_file", default=None, help="Save output to file."
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cuda", "mps", "cpu"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Compute device.",
)
@click.option(
    "--max-tokens", default=32768, show_default=True, help="Max generated tokens."
)
@click.option("--no-timestamps", is_flag=True, help="Omit timestamps in text output.")
@click.option("--no-speakers", is_flag=True, help="Omit speaker IDs in text output.")
@click.option(
    "--model", default=None, help="Model name or path (default depends on engine)."
)
@click.option(
    "-e",
    "--engine",
    type=click.Choice(["vibevoice", "omnilingual"], case_sensitive=False),
    default="vibevoice",
    show_default=True,
    help="Transcription engine.",
)
@click.option(
    "--lang",
    default=None,
    help="Language code for omnilingual engine (e.g. 'eng_Latn', 'ita_Latn').",
)
def main(
    input_source: str,
    context: str | None,
    fmt: str,
    output_file: str | None,
    device: str,
    max_tokens: int,
    no_timestamps: bool,
    no_speakers: bool,
    model: str | None,
    engine: str,
    lang: str | None,
) -> None:
    """Transcribe audio from a file or URL.

    \b
    Examples:
      asr recording.wav
      asr recording.mp3 -f srt -o subtitles.srt
      asr https://youtube.com/watch?v=... -c "VibeVoice, Microsoft"
      asr podcast.m4a --no-speakers --no-timestamps
      asr recording.wav -e omnilingual --lang eng_Latn
    """
    is_omni = engine == "omnilingual"
    model = model or (OMNI_MODEL_DEFAULT if is_omni else MODEL_DEFAULT)

    engine_label = "Omnilingual ASR" if is_omni else "VibeVoice ASR"
    console.print(
        Panel(
            f"[bold bright_cyan]{engine_label}[/] — Speech to Text",
            border_style="bright_cyan",
            padding=(0, 2),
        )
    )

    with tempfile.TemporaryDirectory(prefix="asr_") as tmp_dir:
        # Resolve input
        audio_path = resolve_input(input_source, tmp_dir)
        console.print(f"[green]✓[/] Audio: [bold]{audio_path.name}[/]")

        # Convert unsupported formats (m4a, mp3, webm, etc.) to WAV
        audio_path = ensure_supported_format(audio_path, tmp_dir)

        if is_omni:
            # Omnilingual engine — no device/dtype/attn setup needed
            console.print(f"[dim]  Model: {model}[/]")
            if lang:
                console.print(f"[dim]  Language: {lang}[/]")
            console.print()

            with console.status(
                "[bold cyan]🔄 Loading model…[/] (this may take a moment on first run)",
                spinner="dots",
            ):
                pipeline = load_model_omni(model)
            console.print("[green]✓[/] Model loaded")

            with console.status("[bold cyan]🎙  Transcribing…", spinner="dots"):
                result = transcribe_omni(pipeline, audio_path, lang)
        else:
            # VibeVoice engine
            if device == "auto":
                device = detect_device()
            dtype = detect_dtype(device)
            attn = detect_attention(device)

            console.print(
                f"[dim]  Device: {device}  •  Dtype: {dtype}  •  Attention: {attn}[/]\n"
            )

            with console.status(
                "[bold cyan]🔄 Loading model…[/] (this may take a moment on first run)",
                spinner="dots",
            ):
                asr_model, processor = load_model(model, device, dtype, attn)
            console.print("[green]✓[/] Model loaded")

            result = transcribe_chunked(
                asr_model, processor, audio_path, device, max_tokens, context, tmp_dir
            )

        console.print("\n[green]✓[/] Transcription complete")

        segments = result["segments"]
        raw_text = result["raw_text"]
        elapsed = result["elapsed"]

        # Format output
        if fmt == "json":
            out = json.dumps(
                segments if segments else {"text": raw_text},
                indent=2,
                ensure_ascii=False,
            )
        elif fmt == "srt":
            out = render_srt(segments) if segments else raw_text
        else:
            out = render_text(
                segments,
                raw_text,
                show_timestamps=not no_timestamps,
                show_speakers=not no_speakers,
            )

        # Output
        if output_file:
            Path(output_file).write_text(out, encoding="utf-8")
            console.print(f"\n[green]✓[/] Saved to [bold]{output_file}[/]")
        elif fmt == "text" and segments:
            display_rich(segments, raw_text, elapsed)
        else:
            console.print()
            console.print(out)
            console.print(f"\n[dim]⏱  Completed in {elapsed:.1f}s[/]")
