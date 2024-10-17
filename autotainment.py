import os
import pydub
import random
from moviepy.editor import *
import math

video_list = []
image_list = []
voice_list = []
audio_list = []

video_path = f"autotainment/video/"
image_path = f"autotainment/image/"
voice_path = f"autotainment/output/audio/"
audio_path = f"autotainment/sfx/"



for i in os.listdir(video_path):
    if i.endswith(".mp4") or i.endswith(".mov"):
        video_list.append(i)

for i in os.listdir(image_path):
    if i.endswith(".png"):
        image_list.append(i)

for i in os.listdir(voice_path):
    if i.endswith(".mp3"):
        voice_list.append(i)

for i in os.listdir(audio_path):
    if i.endswith(".mp3"):
        audio_list.append(i)

for n in range(0, 100):
    voice = AudioFileClip(f"{voice_path}{voice_list[random.randint(0, len(voice_list) - 1)]}")

    num_video = 60
    length_video = 16

    video = []

    print("Generating Videos...")

    video.append(ImageClip("autotainment/bg.png"))
    for i in range(0, num_video):

        try:
            vid = VideoFileClip(f"{video_path}{video_list[random.randint(0, len(video_list) - 1)]}")
            video.append(
                vid
                .subclip(0, random.randint(0, min(math.floor(vid.duration), 8)))
                .set_start(length_video / num_video * i)
                .resize((random.randint(0, 2000), random.randint(0, 4000)))
                .fx(vfx.speedx, (random.random() * 1.5) + 0.5)
            )
        except:
            print("\nThere was an error adding a video\n")
            continue

    num_audio = 40
    length_audio = 3

    audio = []

    print("Generating Audios...")
    for i in range(0, num_audio):
        try:
            top_audio = []
            for i in range(0, 3):
                top_audio.append(audio_list[random.randint(0, len(audio_list) - 1)])

            if random.random() * 1.5 < 1:
                aud = AudioFileClip(f"{audio_path}{top_audio[random.randint(0, len(top_audio) - 1)]}")
                audio.append(
                    aud
                    .subclip(0, min(math.floor(aud.duration), random.randint(0, length_audio)))
                    .set_start(
                        random.choices(
                            [
                            math.floor((random.random() * (length_video / 2) + (length_video / 2)) * 10) / 10,
                            math.floor((random.random() * (length_video / 2)) * 10) / 10
                            ],
                            weights=[2/3,1/3],
                            k = 1
                        )[0]
                    )
                )
            else:
                aud = AudioFileClip(f"{audio_path}{audio_list[random.randint(0, len(audio_list) - 1)]}")
                audio.append(
                    aud
                    .subclip(0, min(math.floor(aud.duration), random.randint(0, length_audio)))
                    .set_start(
                        random.choices(
                            [
                            math.floor((random.random() * (length_video / 2) + (length_video / 2)) * 10) / 10,
                            math.floor((random.random() * (length_video / 2)) * 10) / 10
                            ],
                            weights=[2/3,1/3],
                            k = 1
                        )[0]
                    )
                )
        except:
            print("\n\nThere was an error adding an audio\n\n")

    output = CompositeVideoClip(video)

    output.audio = CompositeAudioClip([voice] + audio)

    output = output.subclip(0, length_video)

    output.write_videofile(f"autotainment/output/video/{random.randint(0, 999999999999)}.mp4")
