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

output_dir = "data_barkAI_large"
os.makedirs(output_dir, exist_ok=True)

commands = [
    "Open Youtube on Brave",
    "Get Me Gmail", 
    "New Word Document"
]


def _create_speech_variations(command: str):
    """Create various speech pattern variations for robust training"""
    variations = []
    
    # Original command
    variations.append(command)
    
    # Add pauses at different positions
    words = command.split()
    if len(words) > 2:
        # Pause after first word
        variations.append(f"{words[0]}... {' '.join(words[1:])}")
        # Pause in middle
        mid = len(words) // 2
        variations.append(f"{' '.join(words[:mid])}... {' '.join(words[mid:])}")
        # Pause before last word
        variations.append(f"{' '.join(words[:-1])}... {words[-1]}")
    
    # Add emphasis variations
    variations.append(command.upper())  # Emphasized version
    variations.append(command.lower())  # Soft version
    
    # Add natural speech patterns
    variations.extend([
        f"Um, {command}",
        f"Uh, {command}",
        f"{command}, please",
        f"{command} now",
        f"{command}!",
        f"{command}."
    ])
    
    # Add speed variations through text manipulation
    slow_version = command.replace(" ", "  ")
    variations.append(slow_version)
    
    return variations

def _apply_audio_augmentations(audio: np.ndarray, sample_rate: int):
    """Apply audio augmentations for robustness - optimized version"""
    augmented_versions = [audio]  # Include original
    
    try:
        # Limit augmentations for speed - only use most effective ones
        # Basic augmentations (fastest)
        quiet_audio = audio * 0.7
        loud_audio = np.clip(audio * 1.3, -1.0, 1.0)
        augmented_versions.extend([quiet_audio, loud_audio])
        
        # Add slight noise for robustness (fast)
        noise_factor = 0.003
        noisy_audio = audio + noise_factor * np.random.randn(len(audio))
        augmented_versions.append(noisy_audio)
        
        # Only use librosa for 1-2 most important augmentations to save time
        if LIBROSA_AVAILABLE and random.random() < 0.5:  # 50% chance to apply librosa
            if random.random() < 0.5:
                # Time stretch (choose one randomly)
                rate = random.choice([random.uniform(0.85, 0.95), random.uniform(1.05, 1.25)])
                time_stretched = librosa.effects.time_stretch(audio, rate=rate)
                augmented_versions.append(time_stretched)
            else:
                # Pitch shift (choose one randomly)
                n_steps = random.choice([-1, 1])
                pitch_shifted = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=n_steps)
                augmented_versions.append(pitch_shifted)
        
    except Exception as e:
        print(f"Audio augmentation error: {e}")
    
    return augmented_versions

print("Loading Bark AI models...")
preload_models()

speakers = [
    "v2/en_speaker_0", "v2/en_speaker_1", "v2/en_speaker_2", 
    "v2/en_speaker_3", "v2/en_speaker_4", "v2/en_speaker_5",
    "v2/en_speaker_6", "v2/en_speaker_7", "v2/en_speaker_8", "v2/en_speaker_9"
]

# Generate 500 samples per command with 50 samples per speaker (10 speakers)
total_samples_per_command = 250
samples_per_speaker = 25

print(f"Generating {samples_per_speaker} samples per speaker for each command...")
print(f"Total samples per command: {total_samples_per_command}")

for command in commands:
    command_label = command.replace(" ", "_")
    command_dir = os.path.join(output_dir, command_label)
    os.makedirs(command_dir, exist_ok=True)
    
    print(f"\nGenerating synthetic speech for: '{command}'")
    
    # Get text variations (pre-compute once)
    text_variations = _create_speech_variations(command)
    print(f"Created {len(text_variations)} text variations")
    
    generated_files = []
    
    with tqdm.tqdm(total=total_samples_per_command, desc="Generating audio") as pbar:
        sample_count = 0
        
        for speaker_idx, speaker in enumerate(speakers):
            # Calculate how many samples this speaker should generate
            speaker_target = min(samples_per_speaker, total_samples_per_command - sample_count)
            
            for sample_idx in range(speaker_target):
                try:
                    # Select random text variation (pre-computed)
                    text_variant = text_variations[sample_idx % len(text_variations)]
                    
                    # Generate base audio with Bark
                    audio_array = generate_audio(text_variant, history_prompt=speaker)
                    
                    # Apply limited augmentations for speed
                    augmented_audios = _apply_audio_augmentations(audio_array, SAMPLE_RATE)
                    
                    # Save only the first few augmented versions to control total count
                    max_augs = min(len(augmented_audios), (total_samples_per_command - sample_count))
                    
                    for aug_idx in range(max_augs):
                        if sample_count >= total_samples_per_command:
                            break
                        
                        final_audio = augmented_audios[aug_idx]
                        
                        # Fast normalization
                        final_audio = np.array(final_audio, dtype=np.float32)
                        max_val = np.max(np.abs(final_audio))
                        if max_val > 0:
                            final_audio = final_audio * (0.9 / max_val)
                        
                        # Save file
                        filename = f"{command_label}_speaker{speaker_idx}_var{sample_count:03d}.wav"
                        filepath = os.path.join(command_dir, filename)
                        
                        write_wav(filepath, SAMPLE_RATE, final_audio)
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
