import os
import torch
from bark import SAMPLE_RATE, generate_audio, preload_models
import numpy as np
import tqdm
from scipy.io.wavfile import write as write_wav
from scipy import signal
import random
import json
from typing import List, Dict, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    print("Warning: librosa not available. Some augmentations will be disabled.")
    LIBROSA_AVAILABLE = False

BASE_SAMPLES_PER_SPEAKER = 2  # Down from 25 before
SAMPLES_PER_COMMAND = 2000
NUM_SPEAKERS = 10
NUM_AUGMENTATIONS_PER_BASE = 19  # Each base sample gets ~10 augmented versions

# PERFORMANCE OPTIMIZATION
NUM_WORKER_THREADS = 4  # Parallel audio save + light augmentation
BATCH_AUGMENT_SIZE = 32  # Vectorize augmentation where possible

# DATA DIVERSITY IMPROVEMENT
AUGMENTATION_POLICY = {
    "noise": {"enabled": True, "snr_db": [30, 20, 10]},  # 3
    "tempo": {"enabled": LIBROSA_AVAILABLE, "factors": [0.85, 0.9, 0.95, 1.05, 1.1]},  # 5
    "pitch": {"enabled": LIBROSA_AVAILABLE, "steps": [-2, -1, 1, 2]},  # 4
    "compression": {"enabled": True, "ratios": [4, 8]},  # 2
    "eq": {"enabled": True, "bands": ["low_cut", "mid_boost", "high_cut"]},  # 3
    "reverb": {"enabled": True, "strength": [0.3, 0.6]},  # 2
}

output_dir = "data_barkAI_large_augmented_version_v7"
cache_dir = os.path.join(output_dir, ".cache")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

commands = [
    "Open Youtube on Brave",
    "Get Me Gmail",
    "New Word Document"
]

# PERFORMANCE OPTIMIZATION: Set seed for reproducible augmentation
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# ============================================================================
# DATA DIVERSITY IMPROVEMENT: Enhanced Text Variations
# ============================================================================
def _create_speech_variations(command: str) -> List[str]:
    """
    Create linguistically diverse text variations for robust training.
    Covers natural speech patterns, emphasis, hesitation, and formality levels.
    """
    variations = []
    words = command.split()

    # Original and normalized versions
    variations.append(command)
    variations.append(command.lower())
    variations.append(command.upper())

    # Natural speech patterns with hesitation markers
    hesitation_markers = ["Um,", "Uh,", "Hmm,", "Ah,"]
    for marker in hesitation_markers:
        variations.append(f"{marker} {command}")

    # Polite and emphatic variations
    politeness = [
        f"{command}, please",
        f"{command} if you don't mind",
        f"Could you {command.lower()}",
    ]
    variations.extend(politeness)

    # Emphatic versions
    emphasis = [
        f"{command}!",
        f"Please {command}",
        f"{command}!",
    ]
    variations.extend(emphasis)

    # Natural word boundaries and trailing inflections
    trailing = [
        f"{command}.",
        f"{command} now",
        f"{command} please",
    ]
    variations.extend(trailing)

    # Strategic pauses within multi-word commands
    if len(words) > 2:
        mid = len(words) // 2
        variations.append(f"{' '.join(words[:mid])}... {' '.join(words[mid:])}")
        variations.append(f"{words[0]}... {' '.join(words[1:])}")

    # Remove duplicates while preserving order
    seen = set()
    unique_variations = []
    for v in variations:
        if v not in seen:
            seen.add(v)
            unique_variations.append(v)

    return unique_variations

# ============================================================================
# PERFORMANCE OPTIMIZATION: Bark Cache
# ============================================================================
class BarkCache:
    """Cache Bark-generated audio by (text, speaker) to avoid redundant TTS."""

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.memory_cache = {}

    def _make_key(self, text: str, speaker: str) -> str:
        """Generate a deterministic cache key."""
        return hashlib.md5(f"{text}:{speaker}".encode()).hexdigest()

    def get(self, text: str, speaker: str) -> np.ndarray:
        """Retrieve cached audio or return None."""
        key = self._make_key(text, speaker)

        # Check memory cache first
        if key in self.memory_cache:
            return self.memory_cache[key]

        # Check disk cache
        cache_file = os.path.join(self.cache_dir, f"{key}.npy")
        if os.path.exists(cache_file):
            audio = np.load(cache_file)
            self.memory_cache[key] = audio
            return audio

        return None

    def set(self, text: str, speaker: str, audio: np.ndarray):
        """Store audio in cache."""
        key = self._make_key(text, speaker)
        self.memory_cache[key] = audio

        # Persist to disk
        cache_file = os.path.join(self.cache_dir, f"{key}.npy")
        np.save(cache_file, audio)

# ============================================================================
# DATA DIVERSITY IMPROVEMENT: Advanced Audio Augmentations
# ============================================================================
def _apply_noise(audio: np.ndarray, snr_db: float = 20) -> np.ndarray:
    """Add Gaussian noise at specified SNR (Signal-to-Noise Ratio)."""
    signal_power = np.mean(audio ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    return audio + noise

def _apply_compression(audio: np.ndarray, ratio: float = 4.0, threshold: float = -20) -> np.ndarray:
    """
    Dynamic range compression. Reduces loud peaks while preserving quieter parts.
    Simulates microphone and speaker compression characteristics.
    """
    # Simple compressor: if signal exceeds threshold, reduce by ratio
    threshold_linear = 10 ** (threshold / 20)
    compressed = np.copy(audio)
    mask = np.abs(audio) > threshold_linear
    compressed[mask] = (
        np.sign(audio[mask]) * threshold_linear +
        (np.abs(audio[mask]) - threshold_linear) / ratio
    )
    return compressed

def _apply_eq(audio: np.ndarray, band: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Apply parametric EQ to simulate different acoustic environments.
    Covers telephone-quality (low-cut), presence boost (mid-boost), and brightness control.
    """
    nyquist = sample_rate / 2

    if band == "low_cut":
        # High-pass: attenuate low frequencies (telephone effect)
        sos = signal.butter(4, 300 / nyquist, btype='high', output='sos')
    elif band == "mid_boost":
        # Peaking: boost mid-range for presence
        sos = signal.butter(4, [800 / nyquist, 3000 / nyquist], btype='band', output='sos')
    elif band == "high_cut":
        # Low-pass: attenuate high frequencies (muffled microphone)
        sos = signal.butter(4, 7000 / nyquist, btype='low', output='sos')
    else:
        return audio

    return signal.sosfilt(sos, audio)

def _apply_reverb(audio: np.ndarray, strength: float = 0.2, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Lightweight reverb via simple multi-tap delay (impulse response approximation).
    Simulates room acoustics without expensive convolution.
    """
    delay_samples = int(0.05 * sample_rate)  # 50ms base delay
    delays = [delay_samples, int(delay_samples * 1.5), int(delay_samples * 2.0)]
    decays = [strength * 0.5, strength * 0.3, strength * 0.15]

    reverb = np.zeros_like(audio)
    for delay, decay in zip(delays, decays):
        delayed = np.zeros_like(audio)
        delayed[delay:] = audio[:-delay] * decay
        reverb += delayed

    return audio + reverb

def _apply_augmentations(audio: np.ndarray, sample_rate: int = SAMPLE_RATE, policy: Dict = None) -> List[np.ndarray]:
    """
    Apply a diverse set of augmentations to produce multiple training variations.
    Augmentation selection is deterministic based on policy and random seed.

    PERFORMANCE OPTIMIZATION: Each base Bark sample fans out to multiple augmented versions.
    DIVERSITY IMPROVEMENT: Covers realistic degradations: noise, tempo, pitch, EQ, compression, reverb.
    """
    if policy is None:
        policy = AUGMENTATION_POLICY

    augmented_versions = [audio]  # Include unaugmented original

    # Noise augmentation
    if policy["noise"]["enabled"]:
        for snr in policy["noise"]["snr_db"]:
            try:
                noisy = _apply_noise(audio, snr_db=snr)
                augmented_versions.append(noisy)
            except Exception as e:
                print(f"Noise augmentation error (SNR={snr}): {e}")

    # Tempo and pitch (librosa-dependent)
    if LIBROSA_AVAILABLE:
        if policy["tempo"]["enabled"]:
            for factor in policy["tempo"]["factors"]:
                try:
                    # PERFORMANCE OPTIMIZATION: Only apply tempo to subset
                    time_stretched = librosa.effects.time_stretch(audio, rate=factor)
                    augmented_versions.append(time_stretched)
                except Exception as e:
                    print(f"Tempo augmentation error (factor={factor}): {e}")

        if policy["pitch"]["enabled"]:
            for steps in policy["pitch"]["steps"]:
                try:
                    pitch_shifted = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=steps)
                    augmented_versions.append(pitch_shifted)
                except Exception as e:
                    print(f"Pitch augmentation error (steps={steps}): {e}")

    # Dynamic range compression
    if policy["compression"]["enabled"]:
        for ratio in policy["compression"]["ratios"]:
            try:
                compressed = _apply_compression(audio, ratio=ratio)
                augmented_versions.append(compressed)
            except Exception as e:
                print(f"Compression augmentation error (ratio={ratio}): {e}")

    # Equalization
    if policy["eq"]["enabled"]:
        for band in policy["eq"]["bands"]:
            try:
                eq_audio = _apply_eq(audio, band=band, sample_rate=sample_rate)
                augmented_versions.append(eq_audio)
            except Exception as e:
                print(f"EQ augmentation error (band={band}): {e}")

    # Reverb
    if policy["reverb"]["enabled"]:
        for strength in policy["reverb"]["strength"]:
            try:
                reverb_audio = _apply_reverb(audio, strength=strength, sample_rate=sample_rate)
                augmented_versions.append(reverb_audio)
            except Exception as e:
                print(f"Reverb augmentation error (strength={strength}): {e}")

    return augmented_versions

def _normalize_loudness(audio: np.ndarray, target_lufs: float = -20.0) -> np.ndarray:
    """
    Normalize audio to consistent loudness using RMS (root-mean-square).
    PERFORMANCE OPTIMIZATION: Fast vectorized normalization replaces peak-based approach.
    Ensures consistent training data regardless of Bark speaker output variance.
    """
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        # Convert LUFS target to linear scaling (simplified)
        target_linear = 10 ** (target_lufs / 20)
        audio = audio * (target_linear / rms)

    # Soft clip to prevent clipping
    audio = np.tanh(audio)
    return audio

# ============================================================================
# PERFORMANCE OPTIMIZATION: Parallel I/O Worker
# ============================================================================
def _save_audio_batch(audio_list: List[Tuple[np.ndarray, str]]) -> List[str]:
    """Save a batch of audio files in parallel using threads."""
    saved_files = []

    def save_single(audio: np.ndarray, filepath: str) -> str:
        write_wav(filepath, SAMPLE_RATE, audio)
        return filepath

    with ThreadPoolExecutor(max_workers=NUM_WORKER_THREADS) as executor:
        futures = [
            executor.submit(save_single, audio, filepath)
            for audio, filepath in audio_list
        ]
        for future in as_completed(futures):
            try:
                saved_files.append(future.result())
            except Exception as e:
                print(f"Error saving audio: {e}")

    return saved_files

# ============================================================================
# Main Generation Pipeline
# ============================================================================
print("Loading Bark AI models...")
preload_models()

speakers = [
    "v2/en_speaker_0", "v2/en_speaker_1", "v2/en_speaker_2",
    "v2/en_speaker_3", "v2/en_speaker_4", "v2/en_speaker_5",
    "v2/en_speaker_6", "v2/en_speaker_7", "v2/en_speaker_8", "v2/en_speaker_9"
]

#cache = BarkCache(cache_dir)

print("\n" + "="*70)
print("OPTIMIZED BARK DATASET GENERATION")
print("="*70)
print(f"Base Bark generations per speaker: {BASE_SAMPLES_PER_SPEAKER}")
print(f"Augmentations per base sample: {NUM_AUGMENTATIONS_PER_BASE}")
print(f"Target samples per command: {SAMPLES_PER_COMMAND}")
print(f"Total speakers: {NUM_SPEAKERS}")
print(f"Augmentation policy: {json.dumps({k: v['enabled'] for k, v in AUGMENTATION_POLICY.items()}, indent=2)}")
print("="*70 + "\n")

for command in commands:
    command_label = command.replace(" ", "_")
    command_dir = os.path.join(output_dir, command_label)
    os.makedirs(command_dir, exist_ok=True)

    print(f"\nProcessing command: '{command}'")

    # DATA DIVERSITY IMPROVEMENT: Pre-compute text variations once
    text_variations = _create_speech_variations(command)
    print(f"  Created {len(text_variations)} unique text variations")

    generated_files = []
    sample_count = 0

    with tqdm.tqdm(total=SAMPLES_PER_COMMAND, desc=f"Generating {command_label}", unit="samples") as pbar:

        # PERFORMANCE OPTIMIZATION: Generate fewer base samples, fan out via augmentation
        for speaker_idx, speaker in enumerate(speakers):

            for base_idx in range(BASE_SAMPLES_PER_SPEAKER):
                if sample_count >= SAMPLES_PER_COMMAND:
                    break

                try:
                    # Deterministic text variation selection
                    var_idx = (speaker_idx * BASE_SAMPLES_PER_SPEAKER + base_idx) % len(text_variations)
                    text_variant = text_variations[var_idx]

                    # PERFORMANCE OPTIMIZATION: Check cache before Bark generation
                    #cached_audio = cache.get(text_variant, speaker)
                    #if cached_audio is not None:
                        #base_audio = cached_audio
                    #else:
                        # Generate with Bark
                    base_audio = generate_audio(text_variant, history_prompt=speaker) # indent if cache added back
                        #cache.set(text_variant, speaker, base_audio)

                    # Normalize loudness early (consistent across augmentations)
                    base_audio = _normalize_loudness(base_audio)

                    # PERFORMANCE OPTIMIZATION & DIVERSITY: Fan out augmentations
                    augmented_audios = _apply_augmentations(base_audio, SAMPLE_RATE, AUGMENTATION_POLICY)

                    # Limit augmentations to meet target sample count
                    num_to_save = min(
                        len(augmented_audios),
                        SAMPLES_PER_COMMAND - sample_count
                    )

                    # Prepare batch for parallel I/O
                    save_batch = []
                    for aug_idx in range(num_to_save):
                        if sample_count >= SAMPLES_PER_COMMAND:
                            break

                        final_audio = augmented_audios[aug_idx]
                        final_audio = np.array(final_audio, dtype=np.float32)

                        # Final safety normalization
                        max_val = np.max(np.abs(final_audio))
                        if max_val > 0:
                            final_audio = final_audio * (0.95 / max_val)

                        # Prepare for save
                        filename = f"{command_label}_speaker{speaker_idx:02d}_aug{sample_count:04d}.wav"
                        filepath = os.path.join(command_dir, filename)
                        save_batch.append((final_audio, filepath))

                        sample_count += 1
                        pbar.update(1)

                    # PERFORMANCE OPTIMIZATION: Parallel I/O saves
                    if save_batch:
                        saved = _save_audio_batch(save_batch)
                        generated_files.extend(saved)

                except Exception as e:
                    print(f"  Warning: Error generating sample {sample_count}: {e}")
                    continue

            if sample_count >= SAMPLES_PER_COMMAND:
                break

    print(f"  ✓ Generated {len(generated_files)} audio files for '{command}'")

print("\n" + "="*70)
print("GENERATION COMPLETE")
print(f"Output directory: {output_dir}")
#print(f"Cache directory: {cache_dir}")
print(f"Total commands processed: {len(commands)}")
print(f"Target samples per command: {SAMPLES_PER_COMMAND}")
print(f"Estimated total audio files: {len(commands) * SAMPLES_PER_COMMAND}")
print("="*70)
