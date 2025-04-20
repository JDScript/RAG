from pathlib import Path
import shutil

import av
import datasets
import imagehash
from tqdm import tqdm

from rag.domain.documents import VideoDocument
from rag.domain.documents import VideoFrameDocument

from .utils import clean_captions
from .utils import extract_video_frames

MIN_FRAME_INTERVAL = 5000  # in milliseconds
MAX_FRAME_INTERVAL = 10000  # in milliseconds
HASH_DIFF_THRESHOLD = 15


def get_caption_mid_frame_index(caption, fps):
    """
    Get the frame index for a caption based on its start and end time.
    """
    start_frame = int((caption["start_ms"] / 1000) * fps)
    end_frame = int((caption["end_ms"] / 1000) * fps)
    return (start_frame + end_frame) // 2


def process_video_frames(video):
    # Initialize variables
    video_id = video["json"]["video_id"]
    title = video["json"]["title"]
    captions = clean_captions(video["json"]["captions"])
    frame_generator = extract_video_frames(video["mp4"])
    meta = next(frame_generator)
    total_frames = meta["frames"]
    fps = meta["fps"]
    height = meta["height"]
    width = meta["width"]

    # Initialize Output
    video_document = VideoDocument(
        video_id=video_id,
        video_title=title,
        video_height=height,
        video_width=width,
        video_fps=fps,
        video_total_frames=total_frames,
        captions=captions,
        merged_caption=" ".join(caption["text"] for caption in captions),
        frame_ids=[],
    )
    frame_documents = []

    print(f"Processing video {video_id} with {total_frames} frames")

    save_path = Path(f"frames/{video_id}")
    if save_path.exists():
        shutil.rmtree(save_path)
    save_path.mkdir(parents=True, exist_ok=False)

    # Process frames
    # Caption-based frame indices
    caption_frame_indices = set()
    for caption in captions:
        mid_index = get_caption_mid_frame_index(caption, fps)
        caption_frame_indices.add(mid_index)

    # For fallback interval sampling
    last_sampled_time = -MAX_FRAME_INTERVAL // 2
    last_frame_hash = None
    current_time_ms = 0
    ms_per_frame = 1000 / fps

    for frame_idx, frame in tqdm(enumerate(frame_generator), total=total_frames):
        assert isinstance(frame, av.VideoFrame)
        current_time_ms = frame_idx * ms_per_frame

        sample_reason = None
        should_sample = False

        if frame_idx in caption_frame_indices:
            sample_reason = "caption"
            should_sample = True
        elif (current_time_ms - last_sampled_time) >= MAX_FRAME_INTERVAL:
            sample_reason = "interval"
            should_sample = True

        if should_sample and (current_time_ms - last_sampled_time) >= MIN_FRAME_INTERVAL:
            current_image = frame.to_image()
            current_hash = imagehash.phash(current_image)

            # Avoid redundant visuals
            if last_frame_hash is None or abs(current_hash - last_frame_hash) > HASH_DIFF_THRESHOLD:
                filename = f"{frame_idx:06d}_{sample_reason}.jpg"
                current_image.save(save_path / filename)
                last_sampled_time = current_time_ms
                last_frame_hash = current_hash

                frame_document = VideoFrameDocument(
                    video_id=video_id,
                    frame_index=frame_idx,
                    frame_image=current_image,
                    frame_timestamp=int(current_time_ms),
                )

                frame_documents.append(frame_document)
                video_document.frame_ids.append(str(frame_document.id))

    print(f"Video {video_id} processed with {len(frame_documents)} frames sampled.")
    return video_document, frame_documents


if __name__ == "__main__":
    dataset = datasets.load_dataset("aegean-ai/ai-lectures-spring-24", split="train")

    for video in dataset:
        video_document, frame_documents = process_video_frames(video)
        VideoDocument.bulk_insert([video_document])
        VideoFrameDocument.bulk_insert(frame_documents)
