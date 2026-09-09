# 视频剪辑：将某一视频切段并保存为新视频
from moviepy import VideoFileClip

input_file = "/Users/cangtao/Desktop/test_1.mp4"
output_file = "/Users/cangtao/Desktop/test_2.mp4"

start_time = 2*60+35   # 1分10秒 = 70秒
end_time = 6*60+28     # 2分30秒 = 150秒

with VideoFileClip(input_file) as video:
    if end_time > video.duration:
        raise ValueError(f"视频只有 {video.duration:.1f} 秒，短于结束时间")

    clip = video.subclipped(start_time, end_time)

    clip.write_videofile(
        output_file,
        codec="libx264",
        audio_codec="aac"
    )

    clip.close()

print(f"剪切完成：{output_file}")
