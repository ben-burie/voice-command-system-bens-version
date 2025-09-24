import os
import subprocess
from moviepy.editor import AudioFileClip
import imageio_ffmpeg as ffmpeg_binaries
import ffmpeg

input_folder = "test_recordings"
output_folder = os.path.join(input_folder, "converted_wav")
os.makedirs(output_folder, exist_ok=True)

def convert_to_wav(input_file, output_file):
    try:
        ffmpeg_path = ffmpeg_binaries.get_ffmpeg_exe()
        (
            ffmpeg
            .input(input_file)
            .output(output_file, format='wav', acodec='pcm_s16le', ac=1, ar='44100')
            .run(cmd=ffmpeg_path, overwrite_output=True)
        )
        print(f"Converted {input_file} to {output_file}")
    except ffmpeg.Error as e:
        print(f"FFmpeg error: {e.stderr.decode()}")


for file in os.listdir(input_folder):
    if file.endswith(".m4a"):
        input_file_path = os.path.join(input_folder, file)
        wav_file_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}.wav")
        convert_to_wav(input_file_path, wav_file_path)
