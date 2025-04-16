from collections.abc import Generator
import io

import av
import PIL
import PIL.Image


def time_str_to_ms(t: str) -> int:
    h, m, s = t.split(":")
    s, ms = s.split(".")
    return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms)


def extract_video_frames(mp4_bytes: bytes) -> Generator[PIL.Image.Image]:
    container = av.open(io.BytesIO(mp4_bytes))
    yield {
        "fps": container.streams.video[0].average_rate,
        "height": container.streams.video[0].height,
        "width": container.streams.video[0].width,
        "duration": container.duration,
        "frames": container.streams.video[0].frames,
    }
    for frame in container.decode(video=0):
        yield frame.to_image()
