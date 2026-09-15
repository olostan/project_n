"""
Project N: Resumable Chunked Clip Upload and Ingestion Routes.
"""

import hashlib
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status

from extraction.demux import MediaDecodeError, demux_clip_bytes
from server.deps import get_analysis_service, get_sse_bus, get_vault, verify_auth_token
from server.schemas.clips import (
    AnalyzeRequest,
    AnalyzeResponse,
    ChunkResponse,
    FinalizeRequest,
    FinalizeResponse,
    UploadInitRequest,
    UploadInitResponse,
)
from server.services.analysis_service import AnalysisService
from server.sse_bus import SSEBus
from storage.vault import VaultManager

router = APIRouter(prefix="/api/v1/clips", tags=["clips"])

# In-memory staging for chunked uploads
_active_uploads: dict[str, dict[str, Any]] = {}


@router.post("/upload/init", response_model=UploadInitResponse)
def init_clip_upload(
    req: UploadInitRequest,
    _token: str = Depends(verify_auth_token),
) -> dict[str, Any]:
    """Initializes an authenticated resumable chunked upload."""
    upload_id = str(uuid.uuid4())
    _active_uploads[upload_id] = {
        "file_name": req.file_name,
        "file_size": req.file_size,
        "sha256": req.sha256,
        "total_chunks": req.total_chunks,
        "chunks": {},
    }
    return {"upload_id": upload_id, "chunk_size": 1048576}


@router.put("/upload/{upload_id}/chunk/{chunk_idx}", response_model=ChunkResponse)
async def upload_clip_chunk(
    upload_id: str,
    chunk_idx: int,
    request: Request,
    _token: str = Depends(verify_auth_token),
) -> dict[str, Any]:
    """Receives a binary chunk for an active upload."""
    if upload_id not in _active_uploads:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Upload session not found."
        )

    chunk_bytes = await request.body()
    _active_uploads[upload_id]["chunks"][chunk_idx] = chunk_bytes
    return {"chunk_index": chunk_idx, "received": True}


@router.post("/upload/{upload_id}/finalize", response_model=FinalizeResponse)
def finalize_clip_upload(
    upload_id: str,
    _req: FinalizeRequest,
    vault: VaultManager = Depends(get_vault),
    sse: SSEBus = Depends(get_sse_bus),
    _token: str = Depends(verify_auth_token),
) -> dict[str, Any]:
    """Assembles chunks, verifies checksum, and encrypts into local vault."""
    if upload_id not in _active_uploads:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Upload session not found."
        )

    session = _active_uploads.pop(upload_id)
    chunks = session["chunks"]

    # Assemble all chunks in order
    assembled = bytearray()
    for idx in range(session["total_chunks"]):
        if idx not in chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing chunk {idx} of {session['total_chunks']}.",
            )
        assembled.extend(chunks[idx])

    # Validate sha256 checksum
    actual_hash = hashlib.sha256(assembled).hexdigest()
    if actual_hash != session["sha256"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="SHA-256 integrity checksum mismatch.",
        )

    # Encrypt and store into local vault with per-clip DEK
    clip_id = str(uuid.uuid4())
    vault.store_clip(clip_id, bytes(assembled))

    # Publish SSE event
    sse.publish(
        "processing_progress",
        {
            "task_id": upload_id,
            "stage": "vault_stored",
            "progress": 0.20,
            "clip_id": clip_id,
        },
    )

    return {"clip_id": clip_id, "sha256": actual_hash, "status": "stored"}


@router.post("/{clip_id}/analyze", response_model=AnalyzeResponse)
def analyze_clip(
    clip_id: str,
    req: AnalyzeRequest,
    background_tasks: BackgroundTasks,
    vault: VaultManager = Depends(get_vault),
    sse: SSEBus = Depends(get_sse_bus),
    analysis: AnalysisService = Depends(get_analysis_service),
    _token: str = Depends(verify_auth_token),
) -> dict[str, Any]:
    """Triggers asynchronous extraction and metric matching task on real decrypted media."""
    if not vault.has_clip(clip_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Clip {clip_id} not found in vault.",
        )

    task_id = str(uuid.uuid4())
    sse.publish(
        "processing_progress",
        {
            "task_id": task_id,
            "clip_id": clip_id,
            "stage": "started",
            "progress": 0.05,
        },
    )

    # Dispatch analysis task asynchronously over real decrypted and demuxed data
    def _run_analysis() -> None:
        try:
            raw_media = vault.retrieve_clip(clip_id)
            audio_pcm, video_frames = demux_clip_bytes(raw_media)
            analysis.analyze_sensory_clip(
                clip_id=clip_id,
                audio_pcm=audio_pcm,
                video_frames=video_frames,
                task_id=task_id,
                antecedent_id=req.antecedent_id,
                antecedent_notes=req.antecedent_notes,
                caregiver_hypothesis=req.caregiver_hypothesis,
                setting=req.setting,
                observer=req.observer,
            )

        except MediaDecodeError as err:
            sse.publish(
                "processing_failed",
                {
                    "task_id": task_id,
                    "clip_id": clip_id,
                    "stage": "demux_failed",
                    "error": str(err),
                },
            )
        except Exception as err:
            sse.publish(
                "processing_failed",
                {
                    "task_id": task_id,
                    "clip_id": clip_id,
                    "stage": "analysis_error",
                    "error": str(err),
                },
            )

    background_tasks.add_task(_run_analysis)

    return {"task_id": task_id, "stream_url": "/api/v1/events/stream"}
