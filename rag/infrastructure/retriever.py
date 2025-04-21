from ..domain.chunks import EmbeddedVideoCaptionChunk
from .embeddings import BGEEmbedding


class ContextRetriever:
    FORMAT = """
Retrieved context from video captions:

{context}

Each entry includes:
- Time Range
- Caption Content
- Video Metadata (title, ID, resolution, FPS, duration)
"""

    def retrieve_context(self, query: str) -> str:
        """
        Retrieve context for the given query.
        :param query: The query to retrieve context for.
        :return: The retrieved context.
        """
        emb = BGEEmbedding().embed_text(query)
        search_result = EmbeddedVideoCaptionChunk.search(emb, limit=3)

        formatted_chunks = []
        for chunk in search_result:
            time_range = f"{chunk.start_ms // 1000}s - {chunk.end_ms // 1000}s"
            duration_s = chunk.video_total_frames / chunk.video_fps if chunk.video_fps else "unknown"
            meta_info = (
                f"Video Title: {chunk.video_title}\n"
                f"Video ID: {chunk.video_id}\n"
                f"Resolution: {chunk.video_width}x{chunk.video_height}\n"
                f"FPS: {chunk.video_fps}\n"
                f"Duration: {duration_s:.2f}s\n"
            )
            formatted_chunks.append(
                f"Time: {time_range}\n"
                f"Content: {chunk.content.strip()}\n"
                f"{meta_info}"
                "---"
            )

        full_context = "\n".join(formatted_chunks)
        return self.FORMAT.format(context=full_context.strip())
