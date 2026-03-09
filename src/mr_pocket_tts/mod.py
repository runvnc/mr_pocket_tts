import os
"""
MindRoot Pocket-TTS Plugin

DEBUG_MODE: Enabled - will exit on errors for debugging
Provides local TTS using Pocket-TTS with streaming support for SIP phone integration.
This is a drop-in replacement for mr_eleven_stream that runs locally without API calls.
"""

import asyncio
import io
import subprocess
import audioop
import sys
import queue
import threading
from typing import AsyncGenerator, Optional, Dict, Any

import torch
import torchaudio

from lib.providers.services import service, service_manager
from lib.providers.hooks import hook
from lib.providers.commands import command
from .audio_pacer import AudioPacer

import logging

# Check MR_DEBUG env variable
MR_DEBUG = os.environ.get('MR_DEBUG', '').lower() in ('1', 'true', 'yes')
LOG_LEVEL = logging.DEBUG if MR_DEBUG else logging.WARNING

logging.getLogger('mr_pocket_tts').setLevel(LOG_LEVEL)

logger = logging.getLogger(__name__)

# Debug log file
DEBUG_LOG_FILE = "/tmp/pocket_tts_debug.log"

# Force debug mode - will crash on errors
DEBUG_CRASH_ON_ERROR = True

def fatal_error(msg):
    """Log error and exit process for debugging."""
    print(f"\n\n*** FATAL POCKET-TTS ERROR: {msg} ***\n\n", file=sys.stderr, flush=True)
    logger.error(f"FATAL: {msg}")
    sys.exit(1)

def debug_log(msg):
    """Write debug message to dedicated log file."""
    # Only write to log file if MR_DEBUG is enabled
    if MR_DEBUG:
        import datetime
        with open(DEBUG_LOG_FILE, 'a') as f:
            f.write(f"{datetime.datetime.now().isoformat()} | {msg}\n")
# Default configuration
DEFAULT_VOICE = os.environ.get('MR_POCKET_TTS_VOICE', 'alba')
DEFAULT_OUTPUT_FORMAT = "ulaw_8000"  # Standard for SIP/telephony
POCKET_TTS_SAMPLE_RATE = 24000  # Pocket-TTS native sample rate
ULAW_SAMPLE_RATE = 8000  # Target for SIP

# Global dictionary to track active speak() calls per log_id
_active_speak_locks: Dict[str, asyncio.Lock] = {}
# Global dictionary to track active AudioPacer instances per log_id (for interrupt support)
_active_pacers: Dict[str, Any] = {}


def _get_local_playback_enabled() -> bool:
    """Check if local playback is enabled (no SIP available)."""
    return service_manager.functions.get('sip_audio_out_chunk', None) is None


def _play_audio_locally(audio_data: bytes, sample_rate: int = ULAW_SAMPLE_RATE) -> None:
    """Play audio data locally using ffplay."""
    try:
        logger.debug("Trying to play audio directly with ffplay")
        
        # For ulaw 8kHz
        cmd = [
            'ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet',
            '-f', 'mulaw', '-ar', str(sample_rate), '-ac', '1', '-i', 'pipe:0'
        ]
        
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        process.communicate(input=audio_data)
        
        if process.returncode == 0:
            logger.debug("Played audio using ffplay")
        else:
            logger.warning(f"ffplay failed with return code {process.returncode}")
    except FileNotFoundError:
        logger.warning("ffplay not available for local playback")
    except Exception as e:
        logger.error(f"Error playing audio locally: {str(e)}")


def _convert_to_ulaw(audio_tensor: torch.Tensor, source_rate: int = POCKET_TTS_SAMPLE_RATE) -> bytes:
    """Convert audio tensor to ulaw 8kHz bytes for SIP.
    
    Args:
        audio_tensor: Audio tensor from Pocket-TTS (24kHz)
        source_rate: Source sample rate (default 24000)
    
    Returns:
        ulaw encoded bytes at 8kHz
    """
    # Ensure tensor is on CPU and 1D
    if audio_tensor.is_cuda:
        audio_tensor = audio_tensor.cpu()
    if audio_tensor.dim() == 2:
        audio_tensor = audio_tensor.squeeze(0)
    
    # Convert to 16-bit PCM
    pcm_data = (audio_tensor * 32767).clamp(-32768, 32767).to(torch.int16)
    pcm_bytes = pcm_data.numpy().tobytes()
    
    # Resample from 24kHz to 8kHz using audioop
    # audioop.ratecv expects (fragment, width, nchannels, inrate, outrate, state)
    resampled, _ = audioop.ratecv(pcm_bytes, 2, 1, source_rate, ULAW_SAMPLE_RATE, None)
    
    # Convert to ulaw
    ulaw_bytes = audioop.lin2ulaw(resampled, 2)
    
    return ulaw_bytes


# Sentinel for end of stream
_END_OF_STREAM = object()

# Preload flag
_preload_done = False
_preload_lock = threading.Lock()
_preload_voice = os.environ.get('MR_POCKET_TTS_PRELOAD_VOICE', '')


def preload_model_and_voice():
    """Preload model and optionally a voice at module load time.
    
    Set MR_POCKET_TTS_PRELOAD=1 to enable.
    Set MR_POCKET_TTS_PRELOAD_VOICE to preload a specific voice (e.g., 'alba').
    """
    global _preload_done
    
    if os.environ.get('MR_POCKET_TTS_PRELOAD', '').lower() not in ('1', 'true', 'yes'):
        return
    
    with _preload_lock:
        if _preload_done:
            return
        
        logger.info("Preloading Pocket-TTS model...")
        streamer = get_streamer()
        streamer._ensure_loaded()
        
        if _preload_voice:
            logger.info(f"Preloading voice: {_preload_voice}")
            streamer._get_voice_state(_preload_voice)
        
        _preload_done = True
        logger.info("Pocket-TTS preload complete")


class PocketTTSStreamer:
    """Handles Pocket-TTS model loading and audio generation."""
    
    def __init__(self, model_path: Optional[str] = None, voices_dir: Optional[str] = None):
        self.model = None
        self.voice_cache: Dict[str, dict] = {}
        self.model_path = model_path or os.environ.get('MR_POCKET_TTS_MODEL_PATH')
        self.voices_dir = voices_dir or os.environ.get('MR_POCKET_TTS_VOICES_DIR')
        self.local_playback_enabled = _get_local_playback_enabled()
        self._loaded = False
        
        # Built-in voices from pocket-tts
        self.builtin_voices = ['alba', 'marius', 'javert', 'jean', 'fantine', 'cosette', 'eponine', 'azelma']
    
    def _ensure_loaded(self):
        """Ensure the model is loaded."""
        if self._loaded and self.model is not None:
            return
        
        try:
            from pocket_tts import TTSModel
            
            logger.info("Loading Pocket-TTS model...")
            
            if self.model_path:
                logger.info(f"Loading model from: {self.model_path}")
                self.model = TTSModel.load_model(variant=self.model_path)
            else:
                logger.info("Loading default model from HuggingFace...")
                self.model = TTSModel.load_model()
            
            self._loaded = True
            logger.info(f"Pocket-TTS model loaded. Device: {self.model.device}, Sample Rate: {self.model.sample_rate}")
            
        except ImportError:
            fatal_error("pocket-tts not found. Install with: pip install pocket-tts")
        except Exception as e:
            fatal_error(f"Failed to load Pocket-TTS model: {e}")
    
    @property
    def sample_rate(self) -> int:
        """Get the model's native sample rate."""
        if self.model:
            return self.model.sample_rate
        return POCKET_TTS_SAMPLE_RATE
    
    def _resolve_voice_path(self, voice_id: str) -> str:
        """Resolve voice ID to actual path or built-in name."""
        print(f"[POCKET-TTS DEBUG] _resolve_voice_path called with: {voice_id}", flush=True)
        
        # Check if it's an absolute path that should exist
        if voice_id.startswith('/'):
            if not os.path.exists(voice_id):
                fatal_error(f"Voice file does not exist: {voice_id}")
            print(f"[POCKET-TTS DEBUG] Using absolute path: {voice_id}", flush=True)
            return voice_id
        
        # Check if it's a built-in voice
        if voice_id.lower() in self.builtin_voices:
            print(f"[POCKET-TTS DEBUG] Using built-in voice: {voice_id.lower()}", flush=True)
            return voice_id.lower()
        
        # Check if it's a HuggingFace URL
        if voice_id.startswith('hf://'):
            print(f"[POCKET-TTS DEBUG] Using HuggingFace URL: {voice_id}", flush=True)
            return voice_id
        
        # Check voices directory
        if self.voices_dir:
            for ext in ('.wav', '.mp3', '.flac', '.safetensors'):
                possible_path = os.path.join(self.voices_dir, voice_id)
                if os.path.exists(possible_path):
                    print(f"[POCKET-TTS DEBUG] Found in voices dir: {possible_path}", flush=True)
                    return os.path.abspath(possible_path)
                if not voice_id.endswith(ext):
                    possible_path = os.path.join(self.voices_dir, voice_id + ext)
                    if os.path.exists(possible_path):
                        print(f"[POCKET-TTS DEBUG] Found in voices dir with ext: {possible_path}", flush=True)
                        return os.path.abspath(possible_path)
        
        # Return as-is, let pocket-tts handle it
        print(f"[POCKET-TTS DEBUG] Returning as-is: {voice_id}", flush=True)
        return voice_id
    
    def _get_voice_state(self, voice_id: str) -> dict:
        """Get or cache voice state for a voice ID."""
        self._ensure_loaded()
        
        resolved = self._resolve_voice_path(voice_id)
        print(f"[POCKET-TTS DEBUG] _get_voice_state: voice_id={voice_id}, resolved={resolved}", flush=True)
        
        if resolved in self.voice_cache:
            print(f"[POCKET-TTS DEBUG] Using cached voice state for: {resolved}", flush=True)
            logger.debug(f"Using cached voice state for: {resolved}")
            return self.voice_cache[resolved]
        
        logger.info(f"Loading voice: {resolved}")
        print(f"[POCKET-TTS DEBUG] Loading voice state for: {resolved}", flush=True)
        try:
            state = self.model.get_state_for_audio_prompt(resolved)
            if state is None:
                fatal_error(f"Voice state is None for '{voice_id}' (resolved: {resolved})")
            print(f"[POCKET-TTS DEBUG] Voice state loaded successfully", flush=True)
            self.voice_cache[resolved] = state
            return state
        except Exception as e:
            fatal_error(f"Failed to load voice '{voice_id}' (resolved: {resolved}): {e}")
    
    async def stream_text_to_speech(
        self,
        text: str,
        voice_id: str = DEFAULT_VOICE,
        **kwargs
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream text-to-speech audio in real-time.
        
        Args:
            text: Text to convert to speech
            voice_id: Voice ID (built-in name, file path, or HuggingFace URL)
        
        Yields:
            bytes: Audio chunks as ulaw 8kHz for SIP compatibility
        """
        try:
            print(f"[POCKET-TTS DEBUG] stream_text_to_speech: text={text[:50]}..., voice_id={voice_id}", flush=True)
            logger.info(f"Starting TTS stream for text: {text[:50]}...")
            
            # Check if SIP is available
            if service_manager.functions.get('sip_audio_out_chunk'):
                self.local_playback_enabled = False
            else:
                self.local_playback_enabled = True
            
            # Get voice state
            voice_state = self._get_voice_state(voice_id)
            
            chunk_count = 0
            local_audio_buffer = b"" if self.local_playback_enabled else None
            
            # Use a queue for true streaming between threads
            chunk_queue = queue.Queue(maxsize=10)  # Small buffer for backpressure
            generation_error = [None]  # Use list to allow modification in nested function
            
            def producer():
                """Run TTS generation in a thread, pushing chunks to queue."""
                try:
                    print(f"[POCKET-TTS DEBUG] Producer thread starting", flush=True)
                    for chunk_tensor in self.model.generate_audio_stream(voice_state, text):
                        # Convert immediately in the producer thread
                        ulaw_chunk = _convert_to_ulaw(chunk_tensor, self.sample_rate)
                        chunk_queue.put(ulaw_chunk)
                        print(f"[POCKET-TTS DEBUG] Producer: queued chunk of {len(ulaw_chunk)} bytes", flush=True)
                    print(f"[POCKET-TTS DEBUG] Producer thread finished normally", flush=True)
                except Exception as e:
                    print(f"[POCKET-TTS DEBUG] Producer thread error: {e}", flush=True)
                    logger.error(f"TTS producer error: {e}")
                    generation_error[0] = e
                finally:
                    chunk_queue.put(_END_OF_STREAM)
            
            # Start producer thread
            producer_thread = threading.Thread(target=producer, daemon=True)
            producer_thread.start()
            
            # Consume chunks as they arrive
            loop = asyncio.get_event_loop()
            while True:
                # Non-blocking get with small timeout, run in executor to not block event loop
                try:
                    chunk = await loop.run_in_executor(
                        None, 
                        lambda: chunk_queue.get(timeout=0.05)
                    )
                except queue.Empty:
                    await asyncio.sleep(0.001)  # Tiny yield to event loop
                    continue
                
                if chunk is _END_OF_STREAM:
                    break
                
                chunk_count += 1
                logger.debug(f"Yielding audio chunk {chunk_count}, size: {len(chunk)} bytes")
                
                if self.local_playback_enabled:
                    local_audio_buffer += chunk
                
                yield chunk
            
            # Check if there was an error in generation
            if generation_error[0] is not None:
                fatal_error(f"TTS generation failed: {generation_error[0]}")
            
            print(f"[POCKET-TTS DEBUG] TTS streaming completed. Total chunks: {chunk_count}", flush=True)
            logger.info(f"TTS streaming completed. Total chunks: {chunk_count}")
            
            if chunk_count == 0:
                fatal_error("TTS generated 0 audio chunks - something is wrong!")
            
            # Play locally if enabled
            if self.local_playback_enabled and local_audio_buffer:
                logger.info("Playing audio locally...")
                await loop.run_in_executor(
                    None,
                    _play_audio_locally,
                    local_audio_buffer,
                    ULAW_SAMPLE_RATE
                )
            
        except Exception as e:
            fatal_error(f"Error in TTS streaming: {str(e)}")


# Global streamer instance
_streamer = None

def get_streamer() -> PocketTTSStreamer:
    """Get or create the global Pocket-TTS streamer instance."""
    global _streamer
    if _streamer is None:
        _streamer = PocketTTSStreamer()
    return _streamer


@service()
async def stream_tts(
    text: str,
    voice_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> AsyncGenerator[bytes, None]:
    """
    Service to stream text-to-speech audio in real-time using Pocket-TTS.
    
    This service is designed for backend use with SIP phone calls and returns
    raw audio bytes that can be streamed directly to audio systems.
    
    Args:
        text: Text to convert to speech
        voice_id: Voice ID (optional, uses default if not provided)
        context: MindRoot context (optional)
        **kwargs: Additional parameters (ignored for compatibility)
    
    Yields:
        bytes: Audio chunks as ulaw 8kHz for SIP compatibility
    
    Example usage:
        async for audio_chunk in stream_tts("Hello, this is a test message"):
            await send_to_phone(audio_chunk)
    
    Environment Variables:
        MR_POCKET_TTS_VOICE: Default voice to use
        MR_POCKET_TTS_MODEL_PATH: Path to model file
        MR_POCKET_TTS_VOICES_DIR: Directory containing custom voice files
    """
    try:
        streamer = get_streamer()
        
        voice_id = voice_id or DEFAULT_VOICE
        
        print(f"[POCKET-TTS DEBUG] stream_tts service: voice_id={voice_id}", flush=True)
        logger.info(f"Starting TTS service for text: {text[:50]}...")
        
        async for chunk in streamer.stream_text_to_speech(
            text=text,
            voice_id=voice_id,
            **kwargs
        ):
            yield chunk
            
    except Exception as e:
        fatal_error(f"Error in stream_tts service: {str(e)}")


@command()
async def speak(
    text: str,
    voice_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Convert text to speech using Pocket-TTS streaming.
    
    This command streams the audio in real-time and is designed for backend
    integration with phone systems, audio pipelines, or other streaming audio consumers.
    
    Args:
        text: Text to convert to speech
        voice_id: Voice ID (optional, uses default if not provided)
        context: MindRoot context (optional)
    
    Returns:
        None
    
    Example:
        { "speak": { "text": "Hello, this is a test message" } }
        { "speak": { "text": "Custom voice test", "voice_id": "alba" } }
    
    Environment Variables:
        MR_POCKET_TTS_VOICE: Default voice to use
        MR_POCKET_TTS_MODEL_PATH: Path to model file
        MR_POCKET_TTS_VOICES_DIR: Directory containing custom voice files
    """
    voiceid = voice_id or DEFAULT_VOICE
    print(f"[POCKET-TTS DEBUG] speak() called: text={text[:50]}..., voice_id={voice_id}", flush=True)
    
    try:
        # Get log_id from context for lock management
        log_id = None
        if context and hasattr(context, 'log_id'):
            log_id = context.log_id
        
        # If we have a log_id, check if speak() is already running for it
        if log_id:
            if log_id not in _active_speak_locks:
                _active_speak_locks[log_id] = asyncio.Lock()
            
            lock = _active_speak_locks[log_id]
            
            if lock.locked():
                logger.warning(f"speak() already running for log_id {log_id}, rejecting concurrent call")
                return "ERROR: Speech already in progress for this conversation. Please wait for it to complete."
            
            await lock.acquire()
        
        chunk_count = 0
        local_playback = _get_local_playback_enabled()
        
        # Try to get voice_id from agent persona
        try:
            agent_data = await service_manager.get_agent_data(context.agent_name)
            persona = agent_data["persona"]
            persona_voice = persona.get("voice_id", DEFAULT_VOICE)
            
            print(f"[POCKET-TTS DEBUG] Got persona voice_id: {persona_voice}", flush=True)
            
            # Check if voice_id looks like an ElevenLabs ID (not valid for Pocket-TTS)
            # ElevenLabs IDs are typically 20+ character alphanumeric strings
            if persona_voice and len(persona_voice) > 15 and persona_voice.isalnum():
                print(f"[POCKET-TTS DEBUG] Detected ElevenLabs ID, using default: {DEFAULT_VOICE}", flush=True)
                logger.warning(f"Persona voice_id '{persona_voice}' appears to be an ElevenLabs ID, using default voice.")
                voiceid = DEFAULT_VOICE
            else:
                voiceid = persona_voice
        except Exception as e:
            print(f"[POCKET-TTS DEBUG] Could not get persona voice_id: {e}", flush=True)
            logger.warning(f"Could not get agent persona voice_id, using default. Error: {str(e)}")
            voiceid = voice_id or DEFAULT_VOICE

        print(f"[POCKET-TTS DEBUG] Final voice_id to use: {voiceid}", flush=True)

        # Check if audio is halted (we're in an interrupted state)
        if not local_playback:
            try:
                is_halted = await service_manager.sip_is_audio_halted(context=context)
                if is_halted:
                    logger.info("SPEAK_DEBUG: Audio halted, skipping speak command")
                    return None
            except Exception as e:
                logger.debug(f"Could not check halt status: {e}")
        
        # Use AudioPacer for proper timing when sending to SIP
        if not local_playback:
            pacer = AudioPacer(sample_rate=ULAW_SAMPLE_RATE)
            
            if log_id:
                _active_pacers[log_id] = pacer
            
            async def send_to_sip(chunk, timestamp=None, context=None):
                """Callback for AudioPacer to send chunks to SIP."""
                try:
                    result = await service_manager.sip_audio_out_chunk(chunk, timestamp=timestamp, context=context)
                    return result
                except Exception as e:
                    logger.error(f"Error sending to SIP: {e}")
                    return False
            
            await pacer.start_pacing(send_to_sip, context)
        
        async for chunk in stream_tts(text=text, voice_id=voiceid, context=context):
            chunk_count += 1

            try:
                if not local_playback:
                    if pacer.interrupted:
                        logger.debug("SPEAK_DEBUG: Pacer interrupted, stopping chunk buffering")
                        break
                    
                    await pacer.add_chunk(chunk)
                    logger.debug(f"SPEAK_DEBUG: Buffered chunk {chunk_count}, size: {len(chunk)} bytes")
            except Exception as e:
                logger.warning(f"Error sending audio chunk to SIP output: {str(e)}. Is SIP enabled?")
                await asyncio.sleep(1)
                return f"Error sending audio chunk to SIP output: {str(e)}"

        if not local_playback:
            pacer.mark_finished()
            
            if not pacer.interrupted:
                logger.debug(f"SPEAK_DEBUG: All {chunk_count} chunks buffered, waiting for pacer to finish...")
                await pacer.wait_until_done()
            
            await pacer.stop()
            
            if log_id and log_id in _active_pacers:
                del _active_pacers[log_id]
            
            if pacer.interrupted:
                logger.info(f"SPEAK_DEBUG: Interrupted after {chunk_count} chunks, {pacer.bytes_sent} bytes sent")
                if chunk_count < 2:
                    return "SYSTEM: WARNING - Command interrupted!\n\n"
                return None
            else:
                logger.info(f"SPEAK_DEBUG: Completed {chunk_count} chunks, {pacer.bytes_sent} bytes sent")
        
        print(f"[POCKET-TTS DEBUG] Speech completed: {len(text)} chars, {chunk_count} chunks", flush=True)
        logger.info(f"Speech streaming completed: {len(text)} characters, {chunk_count} audio chunks")
        return None
        
    except Exception as e:
        fatal_error(f"Error in speak command: {str(e)}")

    finally:
        if log_id and log_id in _active_speak_locks:
            lock = _active_speak_locks[log_id]
            if lock.locked():
                lock.release()


@hook()
async def on_interrupt(context=None):
    """
    Handle interruption signal from the system.
    This is called when the user interrupts the AI (e.g., starts speaking during TTS).
    Cancels any active TTS streams for the current session.
    """
    log_id = None
    if context and hasattr(context, 'log_id'):
        log_id = context.log_id
    
    if not log_id:
        logger.debug("on_interrupt called without log_id, cannot cancel specific stream")
        return
    
    if log_id in _active_pacers:
        pacer = _active_pacers[log_id]
        logger.info(f"on_interrupt: Interrupting TTS stream for session {log_id}")
        pacer.interrupt()


@hook()
async def on_agent_loaded(context=None):
    """
    Hook called when an agent is loaded.
    Triggers model preloading if enabled.
    """
    # Run preload in background thread to not block startup
    if os.environ.get('MR_POCKET_TTS_PRELOAD', '').lower() in ('1', 'true', 'yes'):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, preload_model_and_voice)
