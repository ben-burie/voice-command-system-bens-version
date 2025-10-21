# Command to run: python -m scripts.generate_test_inputs_fast

import os
import random
from bark import SAMPLE_RATE, generate_audio
from scipy.io.wavfile import write as write_wav

def get_random_command():
    commands = [
        "Open_Youtube_On_Brave",
        "Get_Me_Gmail", 
        "New_Word_Document"
    ]
    return random.choice(commands)

def get_random_speaker():
    speakers = [
        "v2/en_speaker_0", "v2/en_speaker_1", "v2/en_speaker_2", 
        "v2/en_speaker_3", "v2/en_speaker_4", "v2/en_speaker_5",
        "v2/en_speaker_6", "v2/en_speaker_7", "v2/en_speaker_8", "v2/en_speaker_9"
    ]
    return random.choice(speakers)

def generate_input_command(test_num, chosen_command):
    """Generate Bark AI audio for a given voice command."""
    os.makedirs("input_file", exist_ok=True)

    # Prepare text
    command_text = chosen_command.replace("_", " ")
    speaker = get_random_speaker()

    print(f"[INFO] Generating Bark audio for command: '{command_text}' ({speaker})")

    try:
        # Generate audio
        audio_array = generate_audio(command_text, history_prompt=speaker)

        # Save as WAV
        filename = f"test_{test_num}_{chosen_command}.wav"
        filepath = os.path.join("input_file", filename)
        write_wav(filepath, SAMPLE_RATE, audio_array)

        print(f"[OK] Saved: {filepath}")
        return filepath
    except Exception as e:
        print(f"[ERROR] Failed to generate audio for {chosen_command}: {e}")
        return None

if __name__ == "__main__":
    from bark import preload_models
    preload_models()
    num_files = 300  # change as needed
    for i in range(num_files):
        chosen_command = get_random_command()
        print(f"\n[{i+1}/{num_files}] Command: {chosen_command}")
        filepath = generate_input_command(i, chosen_command)
        if filepath is None:
            print("[WARN] Skipped due to error.")
