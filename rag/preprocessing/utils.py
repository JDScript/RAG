import io

import av


def time_str_to_ms(t: str) -> int:
    h, m, s = t.split(":")
    s, ms = s.split(".")
    return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms)


def clean_captions(captions: list):
    cleaned_captions = []
    for caption in captions:
        if not cleaned_captions:
            cleaned_captions.append(
                {
                    "start": caption["start"],
                    "end": caption["end"],
                    "start_ms": time_str_to_ms(caption["start"]),
                    "end_ms": time_str_to_ms(caption["end"]),
                    "text": caption["text"].strip(),
                }
            )
            continue

        last_caption = cleaned_captions[-1]

        if caption["text"] == last_caption["text"]:
            cleaned_captions[-1]["end"] = caption["end"]
            cleaned_captions[-1]["end_ms"] = time_str_to_ms(caption["end"])
        elif caption["text"].startswith(last_caption["text"]):
            cleaned_captions.append(
                {
                    "start": caption["start"],
                    "end": caption["end"],
                    "start_ms": time_str_to_ms(caption["start"]),
                    "end_ms": time_str_to_ms(caption["end"]),
                    "text": caption["text"][len(last_caption["text"]) :].strip(),
                }
            )
        else:
            cleaned_captions.append(
                {
                    "start": caption["start"],
                    "end": caption["end"],
                    "start_ms": time_str_to_ms(caption["start"]),
                    "end_ms": time_str_to_ms(caption["end"]),
                    "text": caption["text"].strip(),
                }
            )

    return cleaned_captions


def extract_video_frames(mp4_bytes: bytes):
    container = av.open(io.BytesIO(mp4_bytes))
    yield {
        "fps": container.streams.video[0].average_rate,
        "height": container.streams.video[0].height,
        "width": container.streams.video[0].width,
        "duration": container.duration,
        "frames": container.streams.video[0].frames,
    }
    yield from container.decode(video=0)
