import os

import datasets

# 加载数据集
dataset = datasets.load_dataset("aegean-ai/ai-lectures-spring-24", split="train")

def get_video_clip(video_id: str, start_ms: int, end_ms: int):
    # 在数据集中查找视频
    video_clip = None
    for video in dataset:
        if video["json"]["video_id"] == video_id:
            # 提取视频剪辑
            video_clip = video["mp4"]

    # 检查是否找到视频剪辑
    if video_clip is None:
        raise ValueError(f"Video with ID {video_id} not found in the dataset.")

    # 先将字节内容保存到临时文件
    temp_file = f"./temp/temp_{video_id}.mp4"

    if os.path.exists(temp_file):
        return temp_file

    # 确保temp目录存在
    os.makedirs("./temp", exist_ok=True)

    # 将视频数据写入临时文件
    with open(temp_file, "wb") as f:
        f.write(video_clip)

    return temp_file

    # try:
    #     # 现在用文件路径加载视频
    #     with VideoFileClip(temp_file) as clip:
    #         # 将毫秒转换为秒
    #         start_time = int(start_ms / 1000)
    #         end_time = int(end_ms / 1000)
    #         subclip_filename = f"./temp/{video_id}_clip_{start_ms}-{end_ms}.mp4"
    #         if os.path.exists(subclip_filename):
    #             return subclip_filename


    #         # 从视频中提取子剪辑
    #         subclip = clip.subclip(max(0, start_time), min(end_time, clip.duration))

    #         # 保存提取的子剪辑
    #         subclip.write_videofile(subclip_filename, codec="libx264")

    #         # 确保子剪辑也被关闭
    #         subclip.close()

    #         return subclip_filename
    # finally:
    #     # 清理临时文件
    #     if os.path.exists(temp_file):
    #         os.remove(temp_file)
