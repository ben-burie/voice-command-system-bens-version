import os
import math
import time
import numpy as np
from scipy.io.wavfile import write as write_wav
from bark import SAMPLE_RATE, generate_audio, preload_models
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm

# Optional: enable librosa for a small fraction of variants (can slow things)
USE_LIBROSA = False
try:
    if USE_LIBROSA:
        import librosa
        LIBROSA_AVAILABLE = True
    else:
        LIBROSA_AVAILABLE = False
except Exception:
    LIBROSA_AVAILABLE = False

output_dir = "data_barkAI_large"
os.makedirs(output_dir, exist_ok=True)

commands = [
    "Open Youtube on Brave",
    "Get Me Gmail",
    "New Word Document"
]

speaker = "v2/en_speaker_0"

# Targets
total_samples_per_command = 250
# How many final saved files to create from one expensive Bark generation.
# Bigger -> fewer generate_audio calls -> much faster overall.
copies_per_base = 10

# Thread pool for file writes (overlap IO with generation)
WRITE_WORKERS = 4

def cheap_variants(base_audio: np.ndarray, n_variants: int):
    """Fast numpy-only variants. Very cheap operations."""
    variants = []
    base = base_audio.copy()
    variants.append(base)  # original

    if len(variants) >= n_variants:
        return variants[:n_variants]

    # small gain variations
    variants.append(np.clip(base * 0.9, -1.0, 1.0))
    if len(variants) >= n_variants:
        return variants[:n_variants]

    variants.append(np.clip(base * 1.05, -1.0, 1.0))
    if len(variants) >= n_variants:
        return variants[:n_variants]

    # small gaussian noise (very low amplitude)
    noise = base + 0.0008 * np.random.randn(*base.shape).astype(base.dtype)
    variants.append(np.clip(noise, -1.0, 1.0))
    if len(variants) >= n_variants:
        return variants[:n_variants]

    # slight DC offset jitter
    offset = base + (np.random.rand() * 0.0005 - 0.00025)
    variants.append(np.clip(offset, -1.0, 1.0))
    if len(variants) >= n_variants:
        return variants[:n_variants]

    # optionally a single librosa-based time-stretch if enabled and available
    if LIBROSA_AVAILABLE and np.random.rand() < 0.15:
        try:
            rate = 0.99 + 0.02 * (np.random.rand() - 0.5)
            stretched = librosa.effects.time_stretch(base.astype(np.float32), rate=rate)
            if len(stretched) > len(base):
                stretched = stretched[:len(base)]
            else:
                stretched = np.pad(stretched, (0, len(base) - len(stretched)))
            variants.append(np.clip(stretched.astype(base.dtype), -1.0, 1.0))
        except Exception:
            pass

    # fallback: random small scalings until we have enough
    while len(variants) < n_variants:
        scale = 0.96 + 0.08 * np.random.rand()
        variants.append(np.clip(base * scale, -1.0, 1.0))

    return variants[:n_variants]

def write_wav_task(path, arr):
    try:
        write_wav(path, SAMPLE_RATE, arr)
        return path, None
    except Exception as e:
        return path, e

def main():
    print("Preloading Bark models (this is done once)...")
    preload_models()

    print(f"Generating {total_samples_per_command} samples per command using speaker {speaker}")
    print(f"copies_per_base = {copies_per_base} => ~{math.ceil(total_samples_per_command / copies_per_base)} expensive generate_audio calls per command")

    with ThreadPoolExecutor(max_workers=WRITE_WORKERS) as write_pool:
        for command in commands:
            command_label = command.replace(" ", "_")
            command_dir = os.path.join(output_dir, command_label)
            os.makedirs(command_dir, exist_ok=True)

            generated = 0
            base_needed = math.ceil(total_samples_per_command / copies_per_base)
            print(f"\nCommand: '{command}' -> base generations needed: {base_needed}")

            write_futures = []
            with tqdm.tqdm(total=total_samples_per_command, desc=command_label) as pbar:
                for base_idx in range(base_needed):
                    if generated >= total_samples_per_command:
                        break
                    try:
                        t0 = time.time()
                        audio_array = generate_audio(command, history_prompt=speaker)
                        dt = time.time() - t0
                        base = np.array(audio_array, dtype=np.float32)
                        # normalize once
                        maxv = np.max(np.abs(base)) if base.size else 1.0
                        if maxv > 0:
                            base = base * (0.9 / maxv)

                        variants = cheap_variants(base, copies_per_base)
                        for v in variants:
                            if generated >= total_samples_per_command:
                                break
                            vv = np.array(v, dtype=np.float32)
                            maxvv = np.max(np.abs(vv)) if vv.size else 1.0
                            if maxvv > 0:
                                vv = vv * (0.9 / maxvv)
                            fname = f"{command_label}_speaker0_var{generated:03d}.wav"
                            fpath = os.path.join(command_dir, fname)
                            # schedule write
                            fut = write_pool.submit(write_wav_task, fpath, vv)
                            write_futures.append(fut)
                            generated += 1
                            pbar.update(1)

                        # optional quick status
                        print(f"  base {base_idx+1}/{base_needed} gen_time={dt:.2f}s produced {len(variants)} files")

                    except Exception as e:
                        print(f"Warning: generation failure at base {base_idx+1}: {e}")
                        # continue

                # ensure all writes finished for this command (could also keep global)
                for fut in as_completed(write_futures):
                    path, err = fut.result()
                    if err:
                        print(f"Write failed: {path}: {err}")

            print(f"Finished generating {generated} files for '{command}'")

    print("\nAll done.")

if __name__ == "__main__":
    main()