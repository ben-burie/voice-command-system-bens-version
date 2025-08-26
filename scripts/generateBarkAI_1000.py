import os
import torch
from bark import SAMPLE_RATE, generate_audio, preload_models
import numpy as np
import tqdm
from scipy.io.wavfile import write as write_wav

output_dir = "data_barkAI_large"
os.makedirs(output_dir, exist_ok=True)

commands = [
    "Open Youtube on Brave",
    "Open Gmail on Brave", 
    "Create new Word Document"
]

print("Loading Bark AI models...")
preload_models()

speakers = [
    "v2/en_speaker_0"
]

'''
speakers = [
    "v2/en_speaker_0", "v2/en_speaker_1", "v2/en_speaker_2", 
    "v2/en_speaker_3", "v2/en_speaker_4", "v2/en_speaker_5",
    "v2/en_speaker_6", "v2/en_speaker_7", "v2/en_speaker_8", "v2/en_speaker_9"
]
'''

variations_per_speaker = 50

print(f"Generating {variations_per_speaker} variations for each speaker for each command...")

for command in commands:
    command_label = command.replace(" ", "_")
    command_dir = os.path.join(output_dir, command_label)
    os.makedirs(command_dir, exist_ok=True)
    
    print(f"\nGenerating variations for: {command}")
    
    for speaker_idx, speaker in enumerate(speakers):
        print(f"Using speaker: {speaker}")
        
        with tqdm.tqdm(total=variations_per_speaker) as pbar:
            i = 0
            while i < variations_per_speaker:
                output_path = os.path.join(
                    command_dir, 
                    f"{command_label}_speaker{speaker_idx}_var{i:03d}.wav"
                )
                try:
                    audio_array = generate_audio(command, history_prompt=speaker)
                    write_wav(output_path, SAMPLE_RATE, audio_array)
                    pbar.update(1)
                    i += 1
                    
                except Exception as e:
                    print(f"Error generating variation {i} with speaker {speaker}: {e}")

print(f"\nGenerated variations for each command in {output_dir}")
print(f"Total audio files: {len(commands) * len(speakers) * variations_per_speaker}")