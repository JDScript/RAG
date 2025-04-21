from abc import ABC
from typing import TypeVar

from pydantic import Field

from rag.domain.base.qdrant_document import QdrantBaseDocument

from .types import DataCategory

T = TypeVar("T", bound="EmbeddedChunk")


class EmbeddedChunk(QdrantBaseDocument[T], ABC):
    content: str
    embedding: list[float] | None
    video_id: str
    video_title: str
    video_height: int
    video_width: int
    video_fps: int
    video_total_frames: int
    metadata: dict = Field(default_factory=dict)


class EmbeddedVideoCaptionChunk(EmbeddedChunk["EmbeddedVideoCaptionChunk"]):
    start_ms: int
    end_ms: int

    class Config:
        name = DataCategory.VIDEO
        embedding_size = 768


class EmbeddedVideoFrameChunk(EmbeddedChunk):
    frame_index: int
    frame_timestamp: int
    frame_id: str

    class Config:
        name = DataCategory.VIDEO_FRAME
        embedding_size = 1152
