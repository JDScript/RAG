import re

from deepmultilingualpunctuation import PunctuationModel
from loguru import logger
import spacy
from tqdm import tqdm

from ..domain.chunks import EmbeddedVideoCaptionChunk
from ..domain.chunks import EmbeddedVideoFrameChunk
from ..domain.documents import VideoDocument
from ..domain.documents import VideoFrameDocument
from ..infrastructure.embeddings import BGEEmbedding
from ..infrastructure.embeddings import Siglip2Embedding

punctuation_model = PunctuationModel()
spacy_sentence_split = spacy.load("en_core_web_lg")


def build_word_timeline(captions):
    word_timeline = []
    for cap in captions:
        start = cap["start_ms"]
        end = cap["end_ms"]
        words = cap["text"].split()
        for word in words:
            clean_word = re.sub(r"[.,;:!?-]", "", word.lower().strip())
            if clean_word:
                word_timeline.append(
                    {
                        "word": clean_word,
                        "start_ms": start,
                        "end_ms": end,
                    }
                )
    return word_timeline


def match_sentence_time(sentence_words, word_timeline):
    """
    Finds the first exact match in the word timeline.
    Returns (start_ms, end_ms, match_end_idx) if found, else (None, None, None)
    """
    sentence_len = len(sentence_words)
    timeline_words = [w["word"] for w in word_timeline]

    for i in range(len(timeline_words) - sentence_len + 1):
        if timeline_words[i : i + sentence_len] == sentence_words:
            return (word_timeline[i]["start_ms"], word_timeline[i + sentence_len - 1]["end_ms"], i + sentence_len - 1)

    return None, None, None


# Process all video documents
docs = VideoDocument.bulk_find()
logger.info(f"Fetched {len(docs)} video documents from the database.")

for doc in tqdm(docs, desc="Processing video documents"):
    caption_chunks = []
    captions = doc.captions

    # Restore punctuation to get proper sentence boundaries
    doc.merged_caption = punctuation_model.restore_punctuation(doc.merged_caption)

    # Split into sentences, then remove punctuation to match raw captions
    sentences = [
        re.sub(r"[.,;:!?-]", "", sent.text.strip()) for sent in spacy_sentence_split(doc.merged_caption).sents
    ]

    # Merge sentences that are too short
    merged_sentences = []
    current_sentence = ""
    for sentence in sentences:
        if len(current_sentence.split()) + len(sentence.split()) < 250 or len(sentence.split()) < 10:
            current_sentence += " " + sentence
        else:
            merged_sentences.append(current_sentence.strip())
            current_sentence = sentence
    if current_sentence:
        merged_sentences.append(current_sentence.strip())
    sentences = merged_sentences

    # Build a word-level timeline from the original captions
    word_timeline = build_word_timeline(captions)

    for sentence in sentences:
        sentence_words = [w.lower() for w in sentence.split()]
        start_ms, end_ms, end_idx = match_sentence_time(sentence_words, word_timeline)

        if start_ms is None:
            logger.warning(
                f"Could not align sentence: {sentence} with {' '.join([word['word'] for word in word_timeline[: len(sentence_words)]])}"
            )
            continue

        caption_chunks.append(
            {
                "text": sentence,
                "start_ms": start_ms,
                "end_ms": end_ms,
            }
        )

        word_timeline = word_timeline[end_idx + 1 :]

    embedded_chunks = []
    for caption in tqdm(caption_chunks, desc="Creating caption chunks"):
        embedded_chunk = EmbeddedVideoCaptionChunk(
            video_id=doc.video_id,
            video_title=doc.video_title,
            video_height=doc.video_height,
            video_width=doc.video_width,
            video_fps=doc.video_fps,
            video_total_frames=doc.video_total_frames,
            content=caption["text"],
            start_ms=caption["start_ms"],
            end_ms=caption["end_ms"],
            embedding=BGEEmbedding().embed_text(caption["text"]),
        )
        embedded_chunks.append(embedded_chunk)
    EmbeddedVideoCaptionChunk.bulk_insert(embedded_chunks)


for doc in tqdm(docs, desc="Updating video frames documents"):
    frame_ids = doc.frame_ids
    embedded_chunks = []
    for frame_id in tqdm(frame_ids, desc="Creating frame chunks"):
        frame_doc: VideoFrameDocument = VideoFrameDocument.find(_id=frame_id)
        if frame_doc:
            embedded_chunk = EmbeddedVideoFrameChunk(
                content="",
                video_id=doc.video_id,
                video_title=doc.video_title,
                video_height=doc.video_height,
                video_width=doc.video_width,
                video_fps=doc.video_fps,
                video_total_frames=doc.video_total_frames,
                frame_id=str(frame_doc.id),
                frame_index=frame_doc.frame_index,
                frame_timestamp=frame_doc.frame_timestamp,
                embedding=Siglip2Embedding().embed_image(frame_doc.frame_image),
            )
            embedded_chunks.append(embedded_chunk)
    EmbeddedVideoFrameChunk.bulk_insert(embedded_chunks)
