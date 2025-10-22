# Command to run: python -m scripts.generate_test_inputs_fast
import os
import math
import time
import numpy as np
from scipy.io.wavfile import write as write_wav
from bark import SAMPLE_RATE, generate_audio, preload_models
import tqdm

# Optional librosa for time-stretching (nice-to-have, not required)
try:
    import librosa
    LIBROSA_AVAILABLE = True
except Exception:
    LIBROSA_AVAILABLE = False

# Configuration
output_dir = "data_barkAI_fast"
os.makedirs(output_dir, exist_ok=True)

commands = [
    "Open Youtube on Brave",
    "Get Me Gmail",
    "New Word Document"
]

speaker = "v2/en_speaker_0"
total_samples_per_command = 250

# How many final saved files to create from one expensive Bark generation
copies_per_base = 5  # tune this: fewer generate_audio calls if >1

def cheap_variants(base_audio: np.ndarray, n_variants: int):
    """Return up to n_variants cheap variations derived from base_audio (numpy-level ops)."""
    variants = []
    base = base_audio.copy()
    variants.append(base)  # original

    if len(variants) >= n_variants:
        return variants[:n_variants]

    # quieter / louder
    variants.append(np.clip(base * 0.9, -1.0, 1.0))
    if len(variants) >= n_variants:
        return variants[:n_variants]

    variants.append(np.clip(base * 1.05, -1.0, 1.0))
    if len(variants) >= n_variants:
        return variants[:n_variants]

    # small gaussian noise
    noise = base + 0.001 * np.random.randn(*base.shape).astype(base.dtype)
    variants.append(np.clip(noise, -1.0, 1.0))
    if len(variants) >= n_variants:
        return variants[:n_variants]

    # slight speed change using librosa (if available) -> pad/trim to original length
    if LIBROSA_AVAILABLE:
        try:
            rate = 0.98 + 0.04 * np.random.rand()  # between 0.98 and 1.02
            stretched = librosa.effects.time_stretch(base.astype(np.float32), rate=rate)
            if len(stretched) > len(base):
                stretched = stretched[:len(base)]
            else:
                stretched = np.pad(stretched, (0, max(0, len(base) - len(stretched))))
            variants.append(np.clip(stretched.astype(base.dtype), -1.0, 1.0))
        except Exception:
            pass

    # If still not enough, add small pitch-agnostic random scaling variants
    while len(variants) < n_variants:
        scale = 0.95 + 0.1 * np.random.rand()
        variants.append(np.clip(base * scale, -1.0, 1.0))

    return variants[:n_variants]

def main():
    print("Preloading Bark models...")
    preload_models()

    print(f"Generating {total_samples_per_command} samples per command using speaker: {speaker}")
    for command in commands:
        command_label = command.replace(" ", "_")
        command_dir = os.path.join(output_dir, command_label)
        os.makedirs(command_dir, exist_ok=True)

        generated = 0
        base_generations_needed = math.ceil(total_samples_per_command / copies_per_base)
        print(f"\nCommand: '{command}' -> base generations needed: {base_generations_needed} (copies_per_base={copies_per_base})")

        with tqdm.tqdm(total=total_samples_per_command, desc=command_label) as pbar:
            for base_idx in range(base_generations_needed):
                if generated >= total_samples_per_command:
                    break
                try:
                    t0 = time.time()
                    audio_array = generate_audio(command, history_prompt=speaker)
                    dt = time.time() - t0
                    # convert to numpy float32 in range [-1,1]
                    base = np.array(audio_array, dtype=np.float32)
                    # normalize to avoid clipping
                    max_val = np.max(np.abs(base)) if base.size else 1.0
                    if max_val > 0:
                        base = base * (0.9 / max_val)

                    variants = cheap_variants(base, copies_per_base)
                    for v in variants:
                        if generated >= total_samples_per_command:
                            break
                        # final normalization (safeguard)
                        vv = np.array(v, dtype=np.float32)
                        maxv = np.max(np.abs(vv)) if vv.size else 1.0
                        if maxv > 0:
                            vv = vv * (0.9 / maxv)

                        filename = f"{command_label}_speaker0_var{generated:03d}.wav"
                        filepath = os.path.join(command_dir, filename)
                        # scipy can accept float32 data
                        write_wav(filepath, SAMPLE_RATE, vv)
                        generated += 1
                        pbar.update(1)

                    print(f"  base {base_idx+1}/{base_generations_needed} generated_audio_time={dt:.2f}s -> produced {len(variants)} saved files")
                except Exception as e:
                    print(f"Warning: base generation {base_idx+1} failed: {e}")
                    # continue trying until we reach target

        print(f"Finished: generated {generated} files for '{command}'")

    print("\nAll commands generated.")
    print(f"Output folder: {output_dir}")

if __name__ == "__main__":
    main()