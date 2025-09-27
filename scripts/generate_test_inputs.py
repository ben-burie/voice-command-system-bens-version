# Command to run: python -m scripts.generate_test_inputs

import sys
import os
from pathlib import Path
import time
from scripts.voiceCommandConformer import ConformerVoiceCommandSystem
import random
import csv
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
import numpy as np

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    print("Warning: librosa not available. Audio augmentations will be limited.")
    LIBROSA_AVAILABLE = False

def get_random_command():
    commands = [
        "Open_Youtube_On_Brave",
        "Get_Me_Gmail", 
        "New_Word_Document"
    ]
    return commands[random.randint(0, len(commands) - 1)]

def get_random_speaker():
    speakers = [
    "v2/en_speaker_0", "v2/en_speaker_1", "v2/en_speaker_2", 
    "v2/en_speaker_3", "v2/en_speaker_4", "v2/en_speaker_5",
    "v2/en_speaker_6", "v2/en_speaker_7", "v2/en_speaker_8", "v2/en_speaker_9"
    ]
    return speakers[random.randint(0, len(speakers) - 1)]
    

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

def generate_input_command(testNum, chosen_command):
    # Using Bark AI to generate audio files for commands
    #chosen_command = get_random_command() ###########################################
    command_dir = os.path.join("input_file")
    os.makedirs(command_dir, exist_ok=True)

    command = chosen_command.replace("_", " ")
    text_variations = _create_speech_variations(command)

    speaker = get_random_speaker()

    try:
        print(f"[DEBUG] Generating Bark audio for command: '{command}'")

        # Select random text variation (pre-computed)
        text_variant = text_variations[0 % len(text_variations)]

        print(f"[DEBUG] Text variant: {text_variant}")
        
        # Generate base audio with Bark
        audio_array = generate_audio(text_variant, history_prompt=speaker)
        
        print(f"[DEBUG] Bark audio_array type={type(audio_array)}, shape={getattr(audio_array, 'shape', None)}")

        # Apply limited augmentations for speed
        augmented_audios = _apply_audio_augmentations(audio_array, SAMPLE_RATE)

        print(f"[DEBUG] Augmented audios type={type(augmented_audios)}, len={len(augmented_audios)}")
        
        # Save only the first few augmented versions to control total count
        max_augs = 1
        
        for aug_idx in range(max_augs):
            
            final_audio = augmented_audios[aug_idx]

            print(f"[DEBUG] Final audio shape={final_audio.shape}, dtype={final_audio.dtype}")
            
            # Fast normalization
            final_audio = np.array(final_audio, dtype=np.float32)
            max_val = np.max(np.abs(final_audio))
            if max_val > 0:
                final_audio = final_audio * (0.9 / max_val)
            
            # Save file
            filename = "test_" + str(testNum) + "_" + chosen_command + ".wav"
            filepath = os.path.join(command_dir, filename)
            
            #write_wav(filepath, SAMPLE_RATE, final_audio)
            write_wav(filepath, SAMPLE_RATE, (final_audio * 32767).astype(np.int16))
    except Exception as e:
        print(f"[ERROR] Bark generation failed: {e}")
        raise   # re-raise so we see the traceback

    return chosen_command, filepath

if __name__ == "__main__":
    
    chosen_command = get_random_command()
    print(f"Chosen command: {chosen_command}")

    for i in range(500): # Edit this number based on how many test files you want
        print("Generating input command audio...")
        chosen_command, audio_input = generate_input_command(i, chosen_command)
        assert os.path.exists(audio_input), f"Audio input file {audio_input} does not exist."
        print(f"Generated audio file: {audio_input}")
