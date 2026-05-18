from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from typing import Dict, Any
import asyncio
import logging
import os
import shutil
import uuid

from src.audio_downloader_and_processor import process_source_input
from src.audio_transcriber import transcribe_chunked_audio
from src.audio_translator import translate_chunked_audio
from src.rag_pipeline import rag_engine
from src.transcript_summarizer import summarize_transcript
import warnings 
warnings.filterwarnings("ignore", category=SyntaxWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("anuvadak")

app = FastAPI(title="Anuvadak - AI Video Assistant", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_SESSIONS: Dict[str, Dict[str, Any]] = {}
_SESSIONS_LOCK = asyncio.Lock()


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_upload_file(upload: UploadFile, upload_dir: str) -> str:
    _ensure_dir(upload_dir)
    original_name = os.path.basename(upload.filename or "upload")
    unique_name = f"{uuid.uuid4().hex}_{original_name}"
    path = os.path.join(upload_dir, unique_name)

    try:
        with open(path, "wb") as output_file:
            shutil.copyfileobj(upload.file, output_file)
    finally:
        try:
            upload.file.close()
        except Exception:
            pass

    return path


def _cleanup_file(path: str) -> None:
    if path and os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            pass


def _cleanup_dir(path: str) -> None:
    if path and os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


class ChatRequest(BaseModel):
    groq_api_key: str = Field(..., min_length=1, alias="GROQ_API_KEY")
    session_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)

    model_config = {"populate_by_name": True}


@app.post("/summary")
async def summarize_media(
    groq_api_key: str = Form(..., alias="GROQ_API_KEY"),
    mistral_api_key: str = Form(..., alias="MISTRAL_API_KEY"),
    mode: str = Form(...),
    file: UploadFile = File(...),
):
    groq_api_key = groq_api_key.strip()
    mistral_api_key = mistral_api_key.strip()
    mode = mode.strip().lower()

    if not groq_api_key:
        raise HTTPException(status_code=400, detail="GROQ_API_KEY is required.")
    if not mistral_api_key:
        raise HTTPException(status_code=400, detail="MISTRAL_API_KEY is required.")
    if mode not in {"transcribe", "translate"}:
        raise HTTPException(status_code=400, detail="Mode must be 'transcribe' or 'translate'.")

    upload_path = None
    download_dir = None

    try:
        upload_path = _save_upload_file(file, upload_dir="uploads")
        download_dir = os.path.join("downloads", f"job_{uuid.uuid4().hex}")

        def _transcribe_sync() -> str:
            chunk_paths = process_source_input(upload_path, download_dir=download_dir)

            if mode == "transcribe":
                transcript = transcribe_chunked_audio(chunk_paths, groq_api_key)
            else:
                transcript = translate_chunked_audio(chunk_paths, groq_api_key)

            if not transcript or not transcript.strip():
                raise ValueError("No transcript produced.")

            return transcript

        try:
            transcript = await run_in_threadpool(_transcribe_sync)
        except ValueError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Processing failed: {exc}",
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Transcription/translation failed")
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error while processing media: {exc}",
            )

        session_id = uuid.uuid4().hex
        async with _SESSIONS_LOCK:
            _SESSIONS[session_id] = {
                "transcript": transcript,
                "mode": mode,
            }

        try:
            summary_result = await run_in_threadpool(
                lambda: summarize_transcript(transcript, mistral_api_key)
            )
        except Exception as exc:
            logger.exception("Summarization failed")
            return {
                "session_id": session_id,
                "transcript": transcript,
                "summary_status": "failed",
                "summary_error": str(exc),
                "message": "Summarization failed. Use /chat with this session_id for Q&A.",
            }

        return {
            "session_id": session_id,
            "summary_status": "ok",
            "summary": summary_result.get("summary", ""),
            "action_items": summary_result.get("action_items", ""),
            "key_decisions": summary_result.get("key_decisions", ""),
            "open_questions": summary_result.get("open_questions", ""),
        }
    finally:
        _cleanup_file(upload_path)
        _cleanup_dir(download_dir)


@app.post("/chat")
async def chat_with_transcript(payload: ChatRequest):
    async with _SESSIONS_LOCK:
        session_data = _SESSIONS.get(payload.session_id)

    if not session_data:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Run /summary to create one.",
        )

    def _chat_sync():
        rag_chain = rag_engine(session_data["transcript"], payload.groq_api_key)
        result = rag_chain.invoke({"input": payload.query})

        if isinstance(result, dict):
            answer = (
                result.get("answer")
                or result.get("result")
                or result.get("output_text")
            )
        else:
            answer = None

        return answer if answer is not None else str(result)

    try:
        answer = await run_in_threadpool(_chat_sync)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Chat processing failed")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error while answering the question: {exc}",
        )

    return {
        "session_id": payload.session_id,
        "answer": answer,
    }
