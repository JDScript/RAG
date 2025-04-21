from abc import ABC
from typing import TypeVar

from pydantic import Field

from rag.domain.base.qdrant_document import QdrantBaseDocument
from rag.domain.documents import VideoFrameDocument

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

    def to_context(self) -> str | bytes:
        return self.content


class EmbeddedVideoCaptionChunk(EmbeddedChunk["EmbeddedVideoCaptionChunk"]):
    start_ms: int
    end_ms: int

    class Config:
        name = DataCategory.VIDEO
        embedding_size = 768

    def to_context(self) -> str:
        start = self.start_ms / 1000
        end = self.end_ms / 1000
        return f"Video: {self.video_title}\nTime: {start:.2f}s - {end:.2f}s\nCaption: {self.content}"


class EmbeddedVideoFrameChunk(EmbeddedChunk["EmbeddedVideoFrameChunk"]):
    frame_index: int
    frame_timestamp: int
    frame_id: str

    class Config:
        name = DataCategory.VIDEO_FRAME
        embedding_size = 1152

    def to_context(self) -> bytes:
        return VideoFrameDocument.find(_id=self.frame_id).frame_image.tobytes()
