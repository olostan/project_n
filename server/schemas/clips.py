"""
Project N: Pydantic v2 Schemas for Clip Upload and Analysis.
"""

from typing import Any

from pydantic import BaseModel, Field


class UploadInitRequest(BaseModel):
    file_name: str = Field(..., description="Original recording filename")
    file_size: int = Field(..., gt=0, description="Total file size in bytes")
    sha256: str = Field(
        ..., min_length=64, max_length=64, description="Hexadecimal SHA-256 checksum"
    )
    total_chunks: int = Field(..., gt=0, description="Total expected chunks")


class UploadInitResponse(BaseModel):
    upload_id: str
    chunk_size: int = 1048576  # 1 MB chunk default


class ChunkResponse(BaseModel):
    chunk_index: int
    received: bool = True


class FinalizeRequest(BaseModel):
    metadata_json: dict[str, Any] = Field(default_factory=dict)


class FinalizeResponse(BaseModel):
    clip_id: str
    sha256: str
    status: str = "stored"


class AnalyzeRequest(BaseModel):
    generate_render: bool = Field(
        default=False, description="Whether to invoke Qwen LLM card renderer"
    )


class AnalyzeResponse(BaseModel):
    task_id: str
    stream_url: str = "/api/v1/events/stream"
