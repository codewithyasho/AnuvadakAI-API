# Anuvadak - AI Video Assistant API

Turn any local audio or video into a clean transcript, English translation, summary, and interactive Q&A.

## Highlights

- Converts local media to MP3 and chunks it for reliable processing
- Transcribes or translates to English using Groq Whisper
- Summarizes with Mistral and extracts action items, key decisions, and open questions
- Q&A over the transcript using a lightweight RAG pipeline
- FastAPI app with session-based chat flow

## How it works

```
Local media -> MP3 -> Chunk -> Groq (transcribe/translate) -> Transcript
                  |-> Mistral summary + insights
                  |-> RAG Q&A (Groq)
```

## Requirements

- Python 3.12+
- FFmpeg installed and available on PATH
- Groq API key (transcription/translation + Q&A)
- Mistral API key (summaries and insights)

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

or

```bash
uv sync
```

1. Verify FFmpeg is available:

```bash
ffmpeg -version
```

## Run the FastAPI app

```bash
uvicorn app:app --reload
```

Open the docs at:

```
http://127.0.0.1:8000/docs
```

## API

### POST /summary (multipart/form-data)

Required fields:

- GROQ_API_KEY
- MISTRAL_API_KEY
- mode: transcribe or translate
- file: local audio/video file

Success response:

```json
{
 "session_id": "...",
 "summary_status": "ok",
 "summary": "...",
 "action_items": "...",
 "key_decisions": "...",
 "open_questions": "..."
}
```

If summarization fails (for example Mistral quota exceeded), you still get a usable session:

```json
{
 "session_id": "...",
 "summary_status": "failed",
 "summary_error": "...",
 "message": "Summarization failed. Use /chat with this session_id for Q&A.",
 "transcript": "..."
}
```

Example curl:

```bash
curl -X POST "http://127.0.0.1:8000/summary" \
 -H "accept: application/json" \
 -H "Content-Type: multipart/form-data" \
 -F "GROQ_API_KEY=YOUR_GROQ_KEY" \
 -F "MISTRAL_API_KEY=YOUR_MISTRAL_KEY" \
 -F "mode=transcribe" \
 -F "file=@/path/to/video.mp4"
```

### POST /chat (JSON)

Required fields:

- GROQ_API_KEY
- session_id
- query

Response:

```json
{
 "session_id": "...",
 "answer": "..."
}
```

Example curl:

```bash
curl -X POST "http://127.0.0.1:8000/chat" \
 -H "Content-Type: application/json" \
 -d '{
  "GROQ_API_KEY": "YOUR_GROQ_KEY",
  "session_id": "SESSION_ID_FROM_SUMMARY",
  "query": "What were the key points?"
 }'
```

## Session behavior

- A new session_id is created for each /summary call.
- Use the same session_id with /chat to continue Q&A for that summary.
- Sessions are stored in memory. They reset when the server restarts.

## Outputs

These files can be created during processing:

- transcript_original.txt (when transcribing)
- translation_english.txt (when translating)
- full_summary.txt (when summarizing)

Temporary files are stored in uploads/ and downloads/ and are cleaned after each request.

## Scaling notes

- The API is async and offloads heavy work to a thread pool.
- For higher concurrency, run multiple workers (uvicorn --workers N).
- For multi-worker sessions, store sessions in Redis or a database instead of memory.

## Troubleshooting

- 500 during summarization: your Mistral key may be invalid or out of quota.
- Client timeout: long files can take minutes; increase your client timeout.
- FFmpeg not found: install FFmpeg and ensure it is on PATH.

## Security notes

- API keys are provided per request and are not read from .env in the API.
- Avoid logging or sharing keys in public.
