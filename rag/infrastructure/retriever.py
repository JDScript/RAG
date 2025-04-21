from ..domain.chunks import EmbeddedVideoCaptionChunk
from ..domain.chunks import EmbeddedVideoFrameChunk
from ..infrastructure.reranker import Reranker
from .embeddings import BGEEmbedding
from .embeddings import Siglip2Embedding


class ContextRetriever:
    def retrieve_context(self, query: str):
        """
        Retrieve context for the given query.
        :param query: The query to retrieve context for.
        :return: The retrieved context.
        """
        emb = BGEEmbedding().embed_text(query)
        img_emb = Siglip2Embedding().embed_text(query)
        caption_chunks = EmbeddedVideoCaptionChunk.search(
            emb,
            limit=5,
        )
        # Rerank the caption chunks
        rerank_mapping = Reranker().re_rank(query, [chunk.content for chunk in caption_chunks])
        caption_chunks = [caption_chunks[i] for i in rerank_mapping]
        highest_mention_video_id = caption_chunks[0].video_id

        image_chunks = EmbeddedVideoFrameChunk.search(
            img_emb, limit=1, query_filter={"must": [{"key": "video_id", "match": {"value": highest_mention_video_id}}]}
        )

        merged_caption_chunks: list[EmbeddedVideoCaptionChunk] = []
        caption_chunks.sort(key=lambda x: x.start_ms)

        for chunk in caption_chunks:
            if merged_caption_chunks:
                merged_caption_chunks[-1].content += "\n" + chunk.content
                merged_caption_chunks[-1].end_ms = chunk.end_ms
            else:
                merged_caption_chunks.append(chunk)

        return caption_chunks, image_chunks
