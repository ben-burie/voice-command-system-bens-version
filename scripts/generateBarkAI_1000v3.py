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

output_dir = "data_barkAI_large_v2"
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

# Generate 500 samples per command with 50 samples per speaker (10 speakers)
total_samples_per_command = 250 # Changed from 250 to 500
samples_per_speaker = 25

print(f"Generating {samples_per_speaker} samples per speaker for each command...")
print(f"Total samples per command: {total_samples_per_command}")

for command in commands:
    command_label = command.replace(" ", "_")
    command_dir = os.path.join(output_dir, command_label)
    os.makedirs(command_dir, exist_ok=True)
    
    print(f"\nGenerating synthetic speech for: '{command}'")
    
    generated_files = []
    
    with tqdm.tqdm(total=total_samples_per_command, desc="Generating audio") as pbar:
        sample_count = 0
        
        for speaker_idx, speaker in enumerate(speakers):
            # Calculate how many samples this speaker should generate
            speaker_target = min(samples_per_speaker, total_samples_per_command - sample_count)
            
            for sample_idx in range(speaker_target):
                try:
                    
                    # Generate base audio with Bark
                    audio_array = generate_audio(command, history_prompt=speaker)
                        
                    # Save file
                    filename = f"{command_label}_speaker{speaker_idx}_var{sample_count:03d}.wav"
                    filepath = os.path.join(command_dir, filename)
                        
                    write_wav(filepath, SAMPLE_RATE, audio_array)
                    generated_files.append(filepath)
                        
                    sample_count += 1
                    pbar.update(1)
                        
                    if sample_count >= total_samples_per_command:
                        break
                
                except Exception as e:
                    print(f"Warning: Error generating sample {sample_count}: {e}")
                    continue
                
                if sample_count >= total_samples_per_command:
                    break
            
            if sample_count >= total_samples_per_command:
                break
    
    print(f"Generated {len(generated_files)} audio files for '{command}'")

print(f"\nGeneration complete!")
print(f"Generated variations for each command in {output_dir}")
print(f"Total audio files: ~{len(commands) * total_samples_per_command}")