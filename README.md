# MindRoot Pocket-TTS Plugin

A drop-in replacement for `mr_eleven_stream` that uses [Pocket-TTS](https://github.com/teddybear082/pocket-tts-openai_streaming_server) locally instead of the ElevenLabs API.

## Features

- **Local TTS**: Runs entirely on your machine, no API calls needed
- **Streaming Support**: Real-time audio streaming for low latency
- **SIP Compatible**: Outputs ulaw 8kHz audio for telephony integration
- **Voice Cloning**: Use custom voice files or built-in voices
- **CPU Friendly**: Works on CPU (GPU optional but provides minimal benefit)

## Installation

```bash
cd /xfiles/plugins_ah/mr_pocket_tts
pip install -e .
```

Or install dependencies manually:

```bash
pip install pocket-tts torch torchaudio scipy numpy soundfile
```

## Configuration

Environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `MR_POCKET_TTS_VOICE` | Default voice to use | `alba` |
| `MR_POCKET_TTS_MODEL_PATH` | Path to model file (optional) | Auto-download from HuggingFace |
| `MR_POCKET_TTS_VOICES_DIR` | Directory containing custom voice files | None |
| `MR_POCKET_TTS_PRELOAD` | Preload model at startup (1/true/yes) | Disabled |
| `MR_POCKET_TTS_PRELOAD_VOICE` | Voice to preload at startup | None |

## Lowest Latency Configuration

To achieve the lowest possible latency:

### 1. Enable Preloading

Set these environment variables before starting MindRoot:

```bash
export MR_POCKET_TTS_PRELOAD=1
export MR_POCKET_TTS_PRELOAD_VOICE=alba  # or your preferred voice
```

This eliminates:
- Model download/load time (~2-5 seconds on first run)
- Voice state computation (~100-200ms per voice)

### 2. Use Built-in Voices

Built-in voices are faster to load than custom files:
- `alba`, `marius`, `javert`, `jean`, `fantine`, `cosette`, `eponine`, `azelma`

### 3. Keep Text Short

Pocket-TTS streams audio as it generates. Shorter text = faster first audio.
For conversational AI, this is usually natural.

### 4. Expected Latency

| Component | Time |
|-----------|------|
| Model load (first time) | 2-5s (eliminated with preload) |
| Voice state (first time) | 100-200ms (eliminated with preload) |
| First audio chunk | ~50ms |
| Generation RTF | ~0.16x (6x faster than real-time) |

**With preloading enabled, expect ~50ms to first audio chunk.**

## Built-in Voices

Pocket-TTS includes these built-in voices:
- `alba` - Female, neutral
- `marius` - Male
- `javert` - Male
- `jean` - Male
- `fantine` - Female
- `cosette` - Female
- `eponine` - Female
- `azelma` - Female

## Custom Voices

You can use custom voice files by:

1. Setting `MR_POCKET_TTS_VOICES_DIR` to a directory containing `.wav`, `.mp3`, or `.flac` files
2. Using a HuggingFace URL: `hf://username/repo/path/to/voice.wav`
3. Using an absolute file path

**Note**: Custom voices require a 5-10 second audio sample for voice cloning.

## Usage

### As a Service

```python
from lib.providers.services import service_manager

# Stream TTS audio
async for chunk in service_manager.stream_tts("Hello, world!", voice_id="alba"):
    # chunk is ulaw 8kHz bytes
    await send_to_phone(chunk)
```

### As a Command

```json
{ "speak": { "text": "Hello, this is a test message" } }
{ "speak": { "text": "Custom voice", "voice_id": "marius" } }
```

## Agent Persona Integration

Set the `voice_id` in your agent's persona to automatically use that voice:

```json
{
  "persona": {
    "voice_id": "alba"
  }
}
```

## Comparison with mr_eleven_stream

| Feature | mr_pocket_tts | mr_eleven_stream |
|---------|---------------|------------------|
| API Required | No | Yes (ElevenLabs) |
| Cost | Free | Per-character |
| Latency (first chunk) | ~50ms | Network + API (~200-500ms) |
| Voice Quality | Good (65% indistinguishable) | Excellent |
| Custom Voices | File-based cloning | ElevenLabs voices |
| Offline | Yes | No |
| GPU Required | No | N/A (cloud) |

## Technical Details

- **Input**: Text string
- **Output**: ulaw 8kHz mono audio (for SIP compatibility)
- **Native Sample Rate**: 24kHz (Pocket-TTS)
- **Conversion**: Automatic resampling (24kHz → 8kHz) and ulaw encoding
- **Streaming**: True streaming via thread queue (not batch-then-yield)

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Text Input     │────▶│  Pocket-TTS      │────▶│  Audio Chunks   │
│                 │     │  (24kHz PCM)     │     │  (24kHz tensor) │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                        ┌──────────────────┐              │
                        │  Resample +      │◀─────────────┘
                        │  ulaw encode     │
                        │  (8kHz ulaw)     │
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                        │  AudioPacer      │
                        │  (real-time)     │
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                        │  SIP Output      │
                        └──────────────────┘
```

## Troubleshooting

### Model Download

On first run, the model will be downloaded from HuggingFace. This may take a few minutes.
Use `MR_POCKET_TTS_PRELOAD=1` to do this at startup instead of first request.

### CUDA/GPU

Pocket-TTS is optimized for CPU. GPU provides minimal benefit due to:
- Small model size (100M params)
- Batch size of 1
- CPU↔GPU transfer overhead

For CPU-only PyTorch (smaller install):

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Audio Playback

For local testing without SIP, install `ffplay` (part of FFmpeg):

```bash
# Ubuntu/Debian
apt install ffmpeg

# macOS
brew install ffmpeg
```

### High Latency

If experiencing high latency:

1. Enable preloading: `MR_POCKET_TTS_PRELOAD=1`
2. Preload your voice: `MR_POCKET_TTS_PRELOAD_VOICE=alba`
3. Check CPU load - other processes may be competing
4. Use shorter text segments for faster first-chunk delivery
