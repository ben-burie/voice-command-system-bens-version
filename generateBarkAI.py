import os
import torch
from bark import SAMPLE_RATE, generate_audio, preload_models

output_dir = "data_barkAI"
os.makedirs(output_dir, exist_ok=True)

commands = [
    "Open Youtube on Brave",
    "Open Gmail on Brave", 
    "Create new Word Document"
]

preload_models()

speakers = ["v2/en_speaker_0", "v2/en_speaker_1", "v2/en_speaker_2", 
            "v2/en_speaker_3", "v2/en_speaker_4", "v2/en_speaker_6", 
            "v2/en_speaker_7", "v2/en_speaker_8", "v2/en_speaker_9"]

def generate_command_audio(text, output_path, speaker_id):
    audio_array = generate_audio(text, history_prompt=speaker_id)
    
    from scipy.io.wavfile import write as write_wav
    write_wav(output_path, SAMPLE_RATE, audio_array)
    print(f"Generated: {output_path}")

for command in commands:
    command_label = command.replace(" ", "_")
    command_dir = os.path.join(output_dir, command_label)
    os.makedirs(command_dir, exist_ok=True)
    
    for i, speaker in enumerate(speakers):
        generate_command_audio(
            command,
            os.path.join(command_dir, f"{command_label}_speaker{i}.wav"),
            speaker
        )
        
        generate_command_audio(
            command + ".",
            os.path.join(command_dir, f"{command_label}_period_speaker{i}.wav"),
            speaker
        )
        
        if "Open" in command:
            alt_command = command.replace("Open", "Launch")
            alt_label = alt_command.replace(" ", "_")
            generate_command_audio(
                alt_command,
                os.path.join(command_dir, f"{alt_label}_speaker{i}.wav"),
                speaker
            )

print(f"Generated voice command audio files in {output_dir}")
print(f"Each command has its own folder with the command as the label")