from abc import ABC
import base64
from io import BytesIO
from typing import Generic
from typing import TypeVar
import uuid

from loguru import logger
import PIL.Image
from pydantic import UUID4
from pydantic import BaseModel
from pydantic import Field
from pymongo import errors

from rag.infrastructure.mongo import connection
from rag.settings import settings

T = TypeVar("T", bound="MongoBaseDocument")

_database = connection.get_database(settings.MONGO_DATABASE_NAME)


class MongoBaseDocument(BaseModel, Generic[T], ABC):
    model_config = {"arbitrary_types_allowed": True}
    id: UUID4 = Field(default_factory=uuid.uuid4)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, self.__class__):
            return False

        return self.id == value.id

    def __hash__(self) -> int:
        return hash(self.id)

    @staticmethod
    def _is_base64_image(s: str) -> bool:
        try:
            decoded_data = base64.b64decode(s)
            if len(decoded_data) < 8:
                return False

            return bool(decoded_data.startswith((b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n")))
        except Exception:
            return False

    @classmethod
    def from_mongo(cls: type[T], data: dict) -> T:
        """Convert "_id" (str object) into "id" (UUID object)."""

        if not data:
            raise ValueError("Data is empty.")

        id = data.pop("_id")

        for key, value in data.items():
            if isinstance(value, str) and cls._is_base64_image(value):
                data[key] = PIL.Image.open(BytesIO(base64.b64decode(value)))

        return cls(**dict(data, id=id))

    def to_mongo(self: T, **kwargs) -> dict:
        """Convert "id" (UUID object) into "_id" (str object)."""
        exclude_unset = kwargs.pop("exclude_unset", False)
        by_alias = kwargs.pop("by_alias", True)

        parsed = self.model_dump(exclude_unset=exclude_unset, by_alias=by_alias, **kwargs)

        if "_id" not in parsed and "id" in parsed:
            parsed["_id"] = str(parsed.pop("id"))

        return parsed

    def model_dump(self: T, **kwargs) -> dict:
        dict_ = super().model_dump(**kwargs)

        for key, value in dict_.items():
            if isinstance(value, uuid.UUID):
                dict_[key] = str(value)
            if isinstance(value, PIL.Image.Image):
                buffered = BytesIO()
                value.save(buffered, format="PNG")
                dict_[key] = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return dict_

    def save(self: T, **kwargs) -> T | None:
        collection = _database[self.get_collection_name()]
        try:
            collection.insert_one(self.to_mongo(**kwargs))

            return self
        except errors.WriteError:
            logger.exception("Failed to insert document.")

            return None

    @classmethod
    def get_or_create(cls: type[T], **filter_options) -> T:
        collection = _database[cls.get_collection_name()]
        try:
            instance = collection.find_one(filter_options)
            if instance:
                return cls.from_mongo(instance)

            new_instance = cls(**filter_options)
            return new_instance.save()
        except errors.OperationFailure:
            logger.exception(f"Failed to retrieve document with filter options: {filter_options}")

            raise

    @classmethod
    def bulk_insert(cls: type[T], documents: list[T], **kwargs) -> bool:
        collection = _database[cls.get_collection_name()]
        try:
            collection.insert_many(doc.to_mongo(**kwargs) for doc in documents)

            return True
        except (errors.WriteError, errors.BulkWriteError):
            logger.error(f"Failed to insert documents of type {cls.__name__}")

            return False

    @classmethod
    def find(cls: type[T], **filter_options) -> T | None:
        collection = _database[cls.get_collection_name()]
        try:
            instance = collection.find_one(filter_options)
            if instance:
                return cls.from_mongo(instance)

            return None
        except errors.OperationFailure:
            logger.error("Failed to retrieve document")

            return None

    @classmethod
    def bulk_find(cls: type[T], **filter_options) -> list[T]:
        collection = _database[cls.get_collection_name()]
        try:
            instances = collection.find(filter_options)
            return [document for instance in instances if (document := cls.from_mongo(instance)) is not None]
        except errors.OperationFailure:
            logger.error("Failed to retrieve documents")

            return []

    @classmethod
    def get_collection_name(cls: type[T]) -> str:
        if not hasattr(cls, "Settings") or not hasattr(cls.Settings, "name"):
            raise Exception("Document should define an Settings configuration class with the name of the collection.")

        return cls.Settings.name
