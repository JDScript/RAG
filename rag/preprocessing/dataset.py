import datasets

dataset = datasets.load_dataset("aegean-ai/ai-lectures-spring-24", split="train")


for video in dataset:
    video_id = video["json"]["video_id"]
    title = video["json"]["title"]
