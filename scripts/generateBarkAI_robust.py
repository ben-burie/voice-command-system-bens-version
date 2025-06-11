import os
import torch
from bark import SAMPLE_RATE, generate_audio, preload_models
import numpy as np
import tqdm
from scipy.io.wavfile import write as write_wav
import librosa
import soundfile as sf
import random

output_dir = "data_barkAI_robust"
os.makedirs(output_dir, exist_ok=True)

commands = [
    "Open Youtube on Brave",
    "Open Gmail on Brave", 
    "Create new Word Document"
]

print("Loading Bark AI models...")
preload_models()

speakers = [
    "v2/en_speaker_0", "v2/en_speaker_1", "v2/en_speaker_2", 
    "v2/en_speaker_3", "v2/en_speaker_4", "v2/en_speaker_5",
    "v2/en_speaker_6", "v2/en_speaker_7", "v2/en_speaker_8", "v2/en_speaker_9"
]

# Enhanced variations for robustness
def create_speech_variations(command):
    """Create various speech pattern variations"""
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
    variations.append(command.upper())  # Shouted version
    variations.append(command.lower())  # Whispered version
    
    # Add filler words and hesitations
    variations.extend([
        f"Um, {command}",
        f"Uh, {command}",
        f"{command}, please",
        f"Please {command}",
        f"Can you {command}",
        f"I want to {command}",
        f"{command} now"
    ])
    
    # Add speed variations through text manipulation
    # Slow speech simulation with extra spaces
    slow_version = command.replace(" ", "  ")
    variations.append(slow_version)
    
    # Fast speech simulation (bark will naturally vary speed)
    variations.append(f"{command}!")
    variations.append(f"{command}.")
    
    return variations

def apply_audio_augmentations(audio, sample_rate):
    """Apply various audio augmentations to simulate real-world conditions"""
    augmented_versions = []
    
    # Original
    augmented_versions.append(audio)
    
    # Speed variations (time stretching)
    # Slow speech (0.7x to 0.9x speed)
    slow_audio = librosa.effects.time_stretch(audio, rate=random.uniform(0.7, 0.9))
    augmented_versions.append(slow_audio)
    
    # Fast speech (1.1x to 1.4x speed)
    fast_audio = librosa.effects.time_stretch(audio, rate=random.uniform(1.1, 1.4))
    augmented_versions.append(fast_audio)
    
    # Pitch variations (keeping duration same)
    # Lower pitch
    low_pitch = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=-2)
    augmented_versions.append(low_pitch)
    
    # Higher pitch
    high_pitch = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=2)
    augmented_versions.append(high_pitch)
    
    # Add slight noise for robustness
    noise_factor = 0.005
    noisy_audio = audio + noise_factor * np.random.randn(len(audio))
    augmented_versions.append(noisy_audio)
    
    # Volume variations
    quiet_audio = audio * 0.5
    augmented_versions.append(quiet_audio)
    
    loud_audio = audio * 1.5
    loud_audio = np.clip(loud_audio, -1.0, 1.0)  # Prevent clipping
    augmented_versions.append(loud_audio)
    
    return augmented_versions

def add_silence_variations(audio, sample_rate):
    """Add different silence patterns to simulate pauses"""
    variations = []
    
    # Original
    variations.append(audio)
    
    # Add silence at beginning
    silence_start = np.zeros(int(0.5 * sample_rate))  # 0.5 second silence
    variations.append(np.concatenate([silence_start, audio]))
    
    # Add silence at end
    silence_end = np.zeros(int(0.5 * sample_rate))
    variations.append(np.concatenate([audio, silence_end]))
    
    # Add silence in middle (split audio and add pause)
    mid_point = len(audio) // 2
    silence_mid = np.zeros(int(0.3 * sample_rate))  # 0.3 second pause
    variations.append(np.concatenate([audio[:mid_point], silence_mid, audio[mid_point:]]))
    
    return variations

variations_per_speaker = 30  # Reduced to manage generation time

print(f"Generating {variations_per_speaker} robust variations for each speaker for each command...")

for command in commands:
    command_label = command.replace(" ", "_")
    command_dir = os.path.join(output_dir, command_label)
    os.makedirs(command_dir, exist_ok=True)
    
    print(f"\nGenerating robust variations for: {command}")
    
    # Get text variations
    text_variations = create_speech_variations(command)
    
    for speaker_idx, speaker in enumerate(speakers):
        print(f"Using speaker: {speaker}")
        
        with tqdm.tqdm(total=variations_per_speaker) as pbar:
            variation_count = 0
            
            while variation_count < variations_per_speaker:
                try:
                    # Select a random text variation
                    text_variant = random.choice(text_variations)
                    
                    # Generate base audio with Bark
                    audio_array = generate_audio(text_variant, history_prompt=speaker)
                    
                    # Apply audio augmentations
                    augmented_audios = apply_audio_augmentations(audio_array, SAMPLE_RATE)
                    
                    # For each augmented version, also try silence variations
                    for aug_idx, aug_audio in enumerate(augmented_audios):
                        if variation_count >= variations_per_speaker:
                            break
                            
                        silence_variations = add_silence_variations(aug_audio, SAMPLE_RATE)
                        
                        for sil_idx, final_audio in enumerate(silence_variations):
                            if variation_count >= variations_per_speaker:
                                break
                                
                            output_path = os.path.join(
                                command_dir, 
                                f"{command_label}_speaker{speaker_idx}_var{variation_count:03d}.wav"
                            )
                            
                            # Ensure audio is in correct format
                            final_audio = np.array(final_audio, dtype=np.float32)
                            
                            # Normalize audio to prevent clipping
                            if np.max(np.abs(final_audio)) > 0:
                                final_audio = final_audio / np.max(np.abs(final_audio)) * 0.9
                            
                            write_wav(output_path, SAMPLE_RATE, final_audio)
                            variation_count += 1
                            pbar.update(1)
                            
                except Exception as e:
                    print(f"Error generating variation {variation_count} with speaker {speaker}: {e}")
                    continue

print(f"\nGenerated robust variations for each command in {output_dir}")
print(f"Total audio files: {len(commands) * len(speakers) * variations_per_speaker}")
print("\nRobust dataset includes:")
print("- Speed variations (slow and fast speech)")
print("- Pause and silence variations")
print("- Pitch variations")
print("- Volume variations")
print("- Noise robustness")
print("- Hesitation and filler word variations")
