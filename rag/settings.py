from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MONGO_DATABASE_URI: str = "mongodb://root:example@localhost:27017"
    MONGO_DATABASE_NAME: str = "rag"

    QDRANT_DATABASE_HOST: str = "localhost"
    QDRANT_DATABASE_PORT: int = 6333


settings = Settings()
