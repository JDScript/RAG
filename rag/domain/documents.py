from .base.mongo_document import MongoBaseDocument
from .types import DataCategory


class VideoFrameDocument(MongoBaseDocument):
    video_id: str
    video_title: str

    frame: str
    subtitle: str
    start_timestamp: float
    end_timestamp: float

    class Settings:
        name = DataCategory.VIDEO_FRAME
