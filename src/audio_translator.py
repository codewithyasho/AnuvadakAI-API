from groq import Groq
import os


def translate_chunked_audio(chunk_paths: list, groq_api_key: str) -> str:
    """
    Translates audio chunks directly to English using Groq's whisper-large-v3.
    Note: The 'translations' endpoint always outputs English.
    """
    client = Groq(
        api_key=groq_api_key,
        timeout=60.0
    )
    full_translation = []
    print(f"\n[INFO] Starting Groq cloud translation (Audio -> English)...")

    for i, path in enumerate(chunk_paths):
        file_name = os.path.basename(path)
        print(f"Translating Chunk {i+1}/{len(chunk_paths)}: {file_name}")

        try:
            with open(path, "rb") as file:

                translation = client.audio.translations.create(
                    file=(file_name, file.read()),
                    model="whisper-large-v3",
                    response_format="verbose_json",
                )

                if translation.text:
                    full_translation.append(translation.text)

        except Exception as e:
            print(f"❌ Error translating {file_name}: {e}")
            continue

    result = " ".join(full_translation)

    with open("translation_english.txt", "w", encoding="utf-8") as f:
        f.write(result)

    return result
