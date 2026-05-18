from pydub import AudioSegment
import os


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def convert_to_mp3(input_path: str, download_dir: str) -> str:
    """Convert any audio/video file to MP3 format using pydub."""
    _ensure_dir(download_dir)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(download_dir, f"{base_name}_converted.mp3")

    audio = AudioSegment.from_file(input_path)
    # Downsampling helps keep file size low without losing transcription accuracy
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="mp3", bitrate="128k")

    return output_path


def chunk_audio(audio_path: str, chunk_minutes: int = 10, download_dir: str = "downloads") -> list:
    """Chunks the MP3 file into smaller segments for Groq."""
    _ensure_dir(download_dir)
    audio = AudioSegment.from_file(audio_path)
    chunk_ms = chunk_minutes * 60 * 1000
    base_name = os.path.splitext(os.path.basename(audio_path))[0]

    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start: start + chunk_ms]
        chunk_path = os.path.join(download_dir, f"{base_name}_chunk_{i}.mp3")
        chunk.export(chunk_path, format="mp3", bitrate="128k")
        chunks.append(chunk_path)

    return chunks


def process_source_input(source: str, download_dir: str = "downloads") -> list:

    print("\n\n[INFO] Detected local file. Converting to MP3...")
    audio_path = convert_to_mp3(source, download_dir)

    print("\n\n[INFO] Chunking audio...")

    chunks = chunk_audio(audio_path, chunk_minutes=10, download_dir=download_dir)
    print(f"\n✅✅ [INFO] Audio ready — {len(chunks)} chunk(s) created.")

    # Cleanup the original long file to save space
    if "_converted.mp3" in audio_path:
        try:
            os.remove(audio_path)
            print(f"\n❌ [INFO] Removed source audio: {audio_path}\n")
            
        except FileNotFoundError:
            pass

    return chunks
