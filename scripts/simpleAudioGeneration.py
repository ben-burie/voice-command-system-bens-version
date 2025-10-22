# Command to run: python -m scripts.generate_test_inputs_fast

import os
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
import numpy as np
import tqdm

# Simple generator: no text variations, no augmentations.
# Produces `total_samples_per_command` samples per command using a single speaker.

output_dir = "data_barkAI_simple"
os.makedirs(output_dir, exist_ok=True)

commands = [
    "Open Youtube on Brave",
    "Get Me Gmail",
    "New Word Document"
]

speaker = "v2/en_speaker_0"
total_samples_per_command = 250

def main():
    print("Loading Bark AI models...")
    preload_models()

    print(f"Generating {total_samples_per_command} samples per command using speaker: {speaker}")
    for command in commands:
        command_label = command.replace(" ", "_")
        command_dir = os.path.join(output_dir, command_label)
        os.makedirs(command_dir, exist_ok=True)

        print(f"\nGenerating for command: '{command}'")
        generated = 0
        with tqdm.tqdm(total=total_samples_per_command, desc=f"{command_label}") as pbar:
            while generated < total_samples_per_command:
                try:
                    audio_array = generate_audio(command, history_prompt=speaker)
                    audio_arr = np.array(audio_array, dtype=np.float32)

                    # simple normalization (safeguard)
                    max_val = np.max(np.abs(audio_arr))
                    if max_val > 0:
                        audio_arr = audio_arr * (0.9 / max_val)

                    filename = f"{command_label}_speaker0_var{generated:03d}.wav"
                    filepath = os.path.join(command_dir, filename)
                    write_wav(filepath, SAMPLE_RATE, audio_arr)

                    generated += 1
                    pbar.update(1)

                except Exception as e:
                    print(f"Warning: failed to generate sample {generated} for '{command}': {e}")
                    # continue attempting until total reached

        print(f"Finished: generated {generated} files for '{command}'")

    print("\nAll commands generated.")
    print(f"Output folder: {output_dir}")

if __name__ == "__main__":
    main()