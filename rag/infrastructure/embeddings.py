from threading import Lock

from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModel
from transformers import AutoProcessor


class Siglip2Embedding:
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_id: str = "google/siglip2-so400m-patch14-384",
    ):
        if not hasattr(self, "initialized"):
            self.model = AutoModel.from_pretrained(model_id, device_map="auto").eval()
            self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
            self.initialized = True

    def embed_image(self, image):
        """
        Embed an image using the model.

        Args:
            image: The image to embed.

        Returns:
            The embedding of the image.
        """
        inputs = self.processor(images=[image], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            return self.model.get_image_features(**inputs).tolist()[0]

    def embed_text(self, text):
        """
        Embed a text using the model.

        Args:
            text: The text to embed.

        Returns:
            The embedding of the text.
        """
        inputs = self.processor(text=[text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            return self.model.get_text_features(**inputs).tolist()[0]

    @property
    def embedding_size(self):
        """
        Get the size of the embedding.

        Returns:
            The size of the embedding.
        """
        vec = self.embed_text("dummy")
        return vec.shape[1]


class BGEEmbedding:
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_id: str = "BAAI/bge-base-en-v1.5",
    ):
        self.model = SentenceTransformer(model_id)

    def embed_text(self, text):
        return self.model.encode(text, convert_to_numpy=True)

    @property
    def embedding_size(self):
        """
        Get the size of the embedding.

        Returns:
            The size of the embedding.
        """
        vec = self.embed_text("dummy")
        return vec.shape[0]


if __name__ == "__main__":
    embedding = Siglip2Embedding()
    # Example usage
    text_embedding = embedding.embed_text("This is a sample text dwe dew dewd dewd dewdw .")
    print(text_embedding.shape)
