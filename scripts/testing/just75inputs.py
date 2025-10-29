import os
import torch
from bark import SAMPLE_RATE, generate_audio, preload_models
import numpy as np
import tqdm
from scipy.io.wavfile import write as write_wav
import random

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    print("Warning: librosa not available. Audio augmentations will be limited.")
    LIBROSA_AVAILABLE = False

output_dir = "data_barkAI_large5"
os.makedirs(output_dir, exist_ok=True)

commands = [
    "Open Youtube on Brave",
    "Get Me Gmail", 
    "New Word Document"
]

print("Loading Bark AI models...")
preload_models()

speakers = [
    "v2/en_speaker_0"
]

total_samples_per_command = 75  # Total samples to generate per command

generated_files = []

for command in commands:

    command_label = command.replace(" ", "_")
    command_dir = os.path.join(output_dir, command_label)
    os.makedirs(command_dir, exist_ok=True)
    
    print(f"\nGenerating synthetic speech for: '{command}'")

    with tqdm.tqdm(total=total_samples_per_command, desc="Generating audio") as pbar:

        final_audio = generate_audio(command, history_prompt="v2/en_speaker_0")

        # Save file
        filename = f"{command_label}_speaker_0_var{sample_count:03d}.wav"
        filepath = os.path.join(command_dir, filename)
        
        write_wav(filepath, SAMPLE_RATE, final_audio)
        generated_files.append(filepath)

        sample_count += 1
        pbar.update(1)

        if sample_count >= total_samples_per_command:
            break