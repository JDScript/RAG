from ..domain.chunks import EmbeddedVideoCaptionChunk
from .embeddings import BGEEmbedding


class ContextRetriever:
    def __init__(
        self,
    ):
        pass

    def retrieve_context(self, query: str) -> str:
        """
        Retrieve context for the given query.
        :param query: The query to retrieve context for.
        :return: The retrieved context.
        """
        emb = BGEEmbedding().embed_text(query)
        search_result = EmbeddedVideoCaptionChunk.search(emb, limit=10)
        for chunk in search_result:
            print(chunk.start_ms, chunk.end_ms, chunk.content)
            print()


if __name__ == "__main__":
    retriever = ContextRetriever()
    retriever.retrieve_context("explain how ResNets work?")
