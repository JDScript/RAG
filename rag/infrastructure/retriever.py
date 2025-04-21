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
        caption_chunks = EmbeddedVideoCaptionChunk.search(emb, limit=10)
        # Rerank the caption chunks
        rerank_mapping = Reranker().re_rank(query, [chunk.video_title + chunk.content for chunk in caption_chunks])
        caption_chunks = [caption_chunks[i] for i in rerank_mapping]
        highest_mention_video_id = caption_chunks[0].video_id

        image_chunks = EmbeddedVideoFrameChunk.search(
            img_emb, limit=1, query_filter={"must": [{"key": "video_id", "match": {"value": highest_mention_video_id}}]}
        )

        merged_caption_chunks: list[EmbeddedVideoCaptionChunk] = []
        merged = False

        while False:
            for chunk in caption_chunks:
                if not merged_caption_chunks:
                    merged_caption_chunks.append(chunk)
                else:
                    last_chunk = merged_caption_chunks[-1]

                    # Case 1: If current chunk can merge with last chunk (10 seconds tolerance)
                    if chunk.start_ms <= last_chunk.end_ms + 10000:  # 10秒容忍度
                        last_chunk.end_ms = max(last_chunk.end_ms, chunk.end_ms)
                        last_chunk.content += f" {chunk.content}"
                        merged = True
                        break

                    # Case 2: If current chunk can merge by prepending to the last chunk (end_ms - 10 seconds tolerance)
                    if chunk.end_ms >= last_chunk.start_ms - 10000:
                        last_chunk.start_ms = min(last_chunk.start_ms, chunk.start_ms)
                        last_chunk.content = f"{chunk.content} {last_chunk.content}"
                        merged = True
                        break

                    # If no merge, just append the chunk
                    merged_caption_chunks.append(chunk)

            # If no merges occurred, exit the loop
            if not merged:
                break
            # Reset merged flag for the next iteration
            merged = False
            caption_chunks = merged_caption_chunks
            merged_caption_chunks = []

        print("Merged caption chunks:", len(caption_chunks))
        return caption_chunks, image_chunks
