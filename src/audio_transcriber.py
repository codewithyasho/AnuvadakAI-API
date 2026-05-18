from groq import Groq
import os


def transcribe_chunked_audio(chunk_paths: list, groq_api_key: str) -> str:
    """
    Transcribes audio chunks using Groq's whisper-large-v3-turbo.
    """
    client = Groq(
        api_key=groq_api_key,
        timeout=60.0
    )
    full_transcript = []
    print(f"\n[INFO] Starting Groq cloud transcription...")

    for i, path in enumerate(chunk_paths):
        file_name = os.path.basename(path)  # Extract just 'chunk_0.mp3'
        print(f"Sending Chunk {i+1}/{len(chunk_paths)} to Groq: {file_name}")

        try:
            with open(path, "rb") as file:
                transcription = client.audio.transcriptions.create(
                    # Use the base file_name here
                    file=(file_name, file.read()),
                    model="whisper-large-v3-turbo",
                    response_format="verbose_json",
                )

                if transcription.text:
                    full_transcript.append(transcription.text)

        except Exception as e:
            print(f"❌ Error transcribing {file_name}: {e}")
            
            continue

    result = " ".join(full_transcript)

    with open("transcript_original.txt", "w", encoding="utf-8") as f:
        f.write(result)

    return result
