# Command to run: python -m scripts.simpleAudioGeneration

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

def generate_input_command(testNum, chosen_command):
    # Using Bark AI to generate audio files for commands
    #chosen_command = get_random_command() ###########################################
    command_dir = os.path.join("input_file_simple")
    os.makedirs(command_dir, exist_ok=True)

    command = chosen_command.replace("_", " ")

    speaker = get_random_speaker()

    try:
        print(f"[DEBUG] Generating Bark audio for command: '{command}'")

        final_audio = generate_audio(command, speaker=speaker)
        
        # Save file
        filename = "test_" + str(testNum) + "_" + chosen_command + ".wav"
        filepath = os.path.join(command_dir, filename)
        
        write_wav(filepath, SAMPLE_RATE, final_audio)
    except Exception as e:
        print(f"[ERROR] Bark generation failed: {e}")
        raise   # re-raise so we see the traceback

    return chosen_command, filepath

if __name__ == "__main__":
    for i in range(1): # Edit this number based on how many test files you want
        chosen_command = get_random_command()
        print(f"Chosen command: {chosen_command}")
        print("Generating input command audio...")
        chosen_command, audio_input = generate_input_command(i, chosen_command)
        assert os.path.exists(audio_input), f"Audio input file {audio_input} does not exist."
        print(f"Generated audio file: {audio_input}")