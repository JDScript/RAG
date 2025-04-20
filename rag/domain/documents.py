import PIL.Image
from pydantic import Field

from .base.mongo_document import MongoBaseDocument
from .types import DataCategory


class VideoFrameDocument(MongoBaseDocument["VideoFrameDocument"]):
    video_id: str
    frame_index: int
    frame_image: PIL.Image.Image
    frame_timestamp: int = Field(default=0)

    class Settings:
        name = DataCategory.VIDEO_FRAME


class VideoDocument(MongoBaseDocument["VideoDocument"]):
    video_id: str
    video_title: str
    video_height: int
    video_width: int
    video_fps: int
    video_total_frames: int

    captions: list
    merged_caption: str

    frame_ids: list[str] = Field(default_factory=list)

    class Settings:
        name = DataCategory.VIDEO
