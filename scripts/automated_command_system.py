import os
import json
import torch
import torchaudio
import numpy as np
import sounddevice as sd
import threading
import time
import queue
from pathlib import Path
import subprocess
import win32com.client
from datetime import datetime
from typing import Dict, List, Callable
import random
import tqdm
from scipy.io.wavfile import write as write_wav

try:
    from bark import SAMPLE_RATE, generate_audio, preload_models
    BARK_AVAILABLE = True
except ImportError:
    print("Warning: Bark AI not available. Install with: pip install git+https://github.com/suno-ai/bark.git")
    BARK_AVAILABLE = False

import librosa

from voiceCommandConformer import ConformerModel, ConformerVoiceCommandSystem

class AutomatedCommandSystem:
    
    def __init__(self, base_data_dir="data_barkAI_large", model_path="models/conformer_best_model.pth", 
                 config_path="config/command_config.json"):
        self.base_data_dir = Path(base_data_dir)
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.temp_recordings = []
        
        self.base_data_dir.mkdir(exist_ok=True)
        self.config_path.parent.mkdir(exist_ok=True)
        
        self.command_config = self._load_command_config()

        #fix
        self.action_registry = {}
        self._reconstruct_actions()
        
        self.bark_loaded = False
        self.speakers = [
            "v2/en_speaker_0", "v2/en_speaker_1", "v2/en_speaker_2", 
            "v2/en_speaker_3", "v2/en_speaker_4", "v2/en_speaker_5",
            "v2/en_speaker_6", "v2/en_speaker_7", "v2/en_speaker_8", "v2/en_speaker_9"
        ]
        
        self.sample_rate = 16000
        self.recording_duration = 4.0
        self.min_samples_per_command = 10
        self.variations_per_speaker = 15
        self.total_samples_per_command = 100
        
        self.retrain_threshold = 3
        self.commands_since_retrain = 0
        self.voice_system = None
        
        print("Automated Command System initialized")
        print(f"Base data directory: {self.base_data_dir}")
        print(f"Model path: {self.model_path}")
        
        if BARK_AVAILABLE:
            print("Bark AI available for audio generation")
        else:
            print("Bark AI not available - manual recording will be used")
    #fix
    def _reconstruct_actions(self):
        """Reconstructs the action registry from the loaded command config."""
        print("Reconstructing actions from config...")
        commands = self.command_config.get("commands", {})
        for command_name, info in commands.items():
            action_info = info.get("action")
            if not action_info:
                continue

            action_type = action_info.get("type")
            
            # Use a default argument in lambda (e.g., p=path) to correctly capture the variable's value
            if action_type == "app":
                path = action_info.get("path")
                if path:
                    self.action_registry[command_name] = lambda p=path: self._open_application(p)
            elif action_type == "website":
                url = action_info.get("url")
                browser = action_info.get("browser", "default")
                if url:
                    self.action_registry[command_name] = lambda u=url, b=browser: self._open_website(u, b)
            elif action_type == "system":
                cmd = action_info.get("command")
                if cmd:
                    self.action_registry[command_name] = lambda c=cmd: self._execute_system_command(c)
            elif action_type == "custom":
                    self.action_registry[command_name] = lambda cn=command_name: print(f"Executing custom action for {cn}")
        
        print(f"Reconstructed {len(self.action_registry)} actions.")
        
    def _load_command_config(self) -> Dict:
        """Load command configuration from file"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            config = {
                "commands": {},
                "last_trained": None,
                "model_version": 1,
                "auto_retrain": True,
                "retrain_threshold": 5
            }
            self._save_command_config(config)
            return config
    
    def _save_command_config(self, config: Dict):
        """Save command configuration to file"""
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def _load_bark_models(self):
        """Load Bark AI models if not already loaded"""
        if not BARK_AVAILABLE:
            raise Exception("Bark AI not available. Please install it first.")
        
        if not self.bark_loaded:
            print("Loading Bark AI models...")
            preload_models()
            self.bark_loaded = True
            print("Bark AI models loaded successfully")
    
    def _create_speech_variations(self, command: str) -> List[str]:
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
    
    def _apply_audio_augmentations(self, audio: np.ndarray, sample_rate: int) -> List[np.ndarray]:
        """Apply audio augmentations for robustness - optimized version"""
        augmented_versions = [audio]  # Include original
        
        try:
            # Basic augmentations (fastest)
            quiet_audio = audio * 0.7
            loud_audio = np.clip(audio * 1.3, -1.0, 1.0)
            augmented_versions.extend([quiet_audio, loud_audio])
            
            # Add slight noise for robustness (fast)
            noise_factor = 0.003
            noisy_audio = audio + noise_factor * np.random.randn(len(audio))
            augmented_versions.append(noisy_audio)
            
            # Limit librosa augmentations for speed - only use most effective ones
            if random.random() < 0.6:  # 60% chance to apply librosa augmentations
                if random.random() < 0.5:
                    # Time stretch (choose one direction randomly)
                    rate = random.choice([random.uniform(0.85, 0.95), random.uniform(1.05, 1.25)])
                    time_stretched = librosa.effects.time_stretch(audio, rate=rate)
                    augmented_versions.append(time_stretched)
                else:
                    # Pitch shift (choose one direction randomly)
                    n_steps = random.choice([-1, 1])
                    pitch_shifted = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=n_steps)
                    augmented_versions.append(pitch_shifted)
            
        except Exception as e:
            print(f"Audio augmentation error: {e}")
        
        return augmented_versions
    
    def generate_bark_audio_samples(self, command_name: str, num_samples: int = None) -> List[str]:
        """
        Generate audio samples using Bark AI
        """
        if not BARK_AVAILABLE:
            print("Bark AI not available - cannot generate audio samples")
            return []
        
        self._load_bark_models()
        
        if num_samples is None:
            num_samples = self.total_samples_per_command
        
        print(f"\nGenerating {num_samples} audio samples using Bark AI for: '{command_name}'")
        print("Creating high-quality synthetic speech with multiple speakers...")
        
        # Create command directory
        command_dir = self.base_data_dir / command_name
        command_dir.mkdir(exist_ok=True)
        
        # Convert command name to natural speech
        natural_command = command_name.replace("_", " ")
        
        # Get text variations
        text_variations = self._create_speech_variations(natural_command)
        
        generated_files = []
        samples_per_speaker = max(1, num_samples // len(self.speakers))
        
        print(f"Generating ~{samples_per_speaker} samples per speaker")
        
        with tqdm.tqdm(total=num_samples, desc="Generating audio") as pbar:
            sample_count = 0
            
            for speaker_idx, speaker in enumerate(self.speakers):
                if sample_count >= num_samples:
                    break
                
                # Calculate target samples for this speaker
                speaker_target = min(samples_per_speaker, num_samples - sample_count)
                
                for sample_idx in range(speaker_target):
                    try:
                        # Use text variations cyclically for consistency
                        text_variant = text_variations[sample_idx % len(text_variations)]
                        
                        # Generate base audio with Bark
                        audio_array = generate_audio(text_variant, history_prompt=speaker)
                        
                        # Apply optimized augmentations
                        augmented_audios = self._apply_audio_augmentations(audio_array, SAMPLE_RATE)
                        
                        # Limit augmented versions to control total count
                        max_augs = min(len(augmented_audios), num_samples - sample_count)
                        
                        # Save augmented versions
                        for aug_idx in range(max_augs):
                            if sample_count >= num_samples:
                                break
                            
                            final_audio = augmented_audios[aug_idx]
                            
                            # Fast normalization
                            final_audio = np.array(final_audio, dtype=np.float32)
                            max_val = np.max(np.abs(final_audio))
                            if max_val > 0:
                                final_audio = final_audio * (0.9 / max_val)
                            
                            # Save file
                            filename = f"{command_name}_speaker{speaker_idx}_var{sample_count:03d}.wav"
                            filepath = command_dir / filename
                            
                            write_wav(str(filepath), SAMPLE_RATE, final_audio)
                            generated_files.append(str(filepath))
                            
                            sample_count += 1
                            pbar.update(1)
                            
                            if sample_count >= num_samples:
                                break
                    
                    except Exception as e:
                        print(f"Warning: Error generating sample {sample_count}: {e}")
                        continue
                    
                    if sample_count >= num_samples:
                        break
                
                if sample_count >= num_samples:
                    break
        
        print(f"Generated {len(generated_files)} audio files using Bark AI")
        return generated_files
    
    #fix
    def add_new_command(self, command_name: str, action_info: Dict, action_function: Callable, 
                       num_samples: int = None, auto_train: bool = True, use_bark: bool = True) -> bool:
        """
        Add a new voice command to the system using Bark AI or manual recording
        
        Args:
            command_name: Name of the command (e.g., "Open_Discord")
            action_function: Function to execute when command is recognized
            num_samples: Number of audio samples to generate/record
            auto_train: Whether to automatically retrain the model
            use_bark: Whether to use Bark AI for generation (True) or manual recording (False)
        """
        try:
            print(f"\n Adding New Command: {command_name}")
            
            # Generate audio samples using Bark AI
            if use_bark and BARK_AVAILABLE:
                print(" Using Bark AI for audio generation...")
                generated_files = self.generate_bark_audio_samples(command_name, num_samples)
                generation_method = "bark_ai"
            else:
                print(" Bark AI not available - cannot add command without audio samples")
                return False
            
            # Update command configuration
            #fix
            self.command_config["commands"][command_name] = {
                "action": action_info,  # Save the serializable dictionary
                "audio_files": generated_files,
                "added_date": datetime.now().isoformat(),
                "sample_count": len(generated_files),
                "generation_method": generation_method
            }
            
            # Save configuration
            self._save_command_config(self.command_config)
            
            # Register the callable action_function for the current session
            self.action_registry[command_name] = action_function
            
            self.commands_since_retrain += 1
            
            print(f" Command '{command_name}' added successfully!")
            print(f" Generated {len(generated_files)} audio samples using {generation_method}")
            
            # Auto-retrain if threshold reached
            if auto_train and self.commands_since_retrain >= self.retrain_threshold:
                print(f"\n Auto-retraining triggered (threshold: {self.retrain_threshold})")
                self.retrain_model()
            
            return True
            
        except Exception as e:
            print(f" Error adding command '{command_name}': {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def retrain_model(self) -> bool:
        """
        Retrain the conformer model with all available commands using adaptive incremental trainer
        """
        try:
            print("\n=== Retraining Conformer Model ===")
            
            # Use the adaptive incremental trainer for full retraining
            from adaptive_incremental_trainer import AdaptiveIncrementalTrainer
            
            trainer = AdaptiveIncrementalTrainer(
                model_path=str(self.model_path),
                base_data_dir=str(self.base_data_dir),
                config_path=str(self.config_path.parent / "adaptive_incremental_config.json")
            )
            
            # Get all current commands for full retraining
            all_commands = []
            for folder in self.base_data_dir.iterdir():
                if folder.is_dir():
                    all_commands.append(folder.name)
            
            # Use incremental training with all commands (effectively full retraining)
            success = trainer.incremental_train(all_commands, epochs=15)
            
            if success:
                # Update configuration
                self.command_config["last_trained"] = datetime.now().isoformat()
                self.command_config["model_version"] += 1
                self._save_command_config(self.command_config)
                
                self.commands_since_retrain = 0
                print(" Model retrained successfully!")
                return True
            else:
                print(" Model retraining failed!")
                return False
            
        except Exception as e:
            print(f" Error retraining model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def reload_voice_system(self):
        """
        Reload the voice command system with the updated model
        """
        try:
            if self.voice_system:
                self.voice_system.stop_listening()
                time.sleep(1)
            
            # Create new voice system with updated model and actions
            self.voice_system = ConformerVoiceCommandSystem(
                str(self.model_path), 
                action_registry=getattr(self, 'action_registry', {})
            )
            
            print(" Voice command system reloaded")
            return True
            
        except Exception as e:
            print(f" Error reloading voice system: {e}")
            return False
    
    def start_voice_system(self):
        """Start the voice command system"""
        if not self.voice_system:
            self.reload_voice_system()
        
        if self.voice_system:
            self.voice_system.start()
    
    def add_command_interactive(self):
        """
        Interactive command addition process
        """
        print("\n=== Interactive Command Addition ===")
        
        command_name = input("Enter command name (e.g., 'Open_Discord'): ").strip()
        if not command_name:
            print(" Command name cannot be empty")
            return
        
        print("\nAvailable action types:")
        print("1. Open application")
        print("2. Open website")
        print("3. System command")
        print("4. Custom action")
        
        action_type = input("Select action type (1-4): ").strip()
        
        action_function = None
        action_info = None
        
        if action_type == "1":
            app_path = input("Enter application path or command: ").strip()
            action_function = lambda: self._open_application(app_path)
            action_info = {"type": "app", "path": app_path}
            
        elif action_type == "2":
            url = input("Enter website URL: ").strip()
            browser = input("Enter browser (brave/chrome/default): ").strip() or "default"
            action_function = lambda: self._open_website(url, browser)
            action_info = {"type": "website", "url": url, "browser": browser}
            
        elif action_type == "3":
            cmd = input("Enter system command: ").strip()
            action_function = lambda: self._execute_system_command(cmd)
            action_info = {"type": "system", "command": cmd}
            
        elif action_type == "4":
            print("Custom action - you'll need to implement this in code")
            action_function = lambda: print(f"Executing custom action for {command_name}")
            action_info = {"type": "custom"}
        
        else:
            print(" Invalid action type")
            return
        
        # Add the command
        num_samples = int(input("Number of audio samples to record (default 10): ") or "10")
        
        success = self.add_new_command(command_name, action_info, action_function, num_samples)
        
        if success:
            print(f" Command '{command_name}' added successfully!")
            
            # Ask if user wants to reload the voice system
            reload = input("Reload voice system now? (y/n): ").strip().lower()
            if reload == 'y':
                self.reload_voice_system()
    
    def _execute_system_command(self, cmd: str):
        """Execute system command"""
        try:
            subprocess.run(cmd, shell=True)
            print(f"Executed: {cmd}")
        except Exception as e:
            print(f"Error executing command: {e}")
        
    def _open_application(self, app_path: str):
        """Open an application"""
        try:
            subprocess.Popen(app_path, shell=True)
            print(f"Opening application: {app_path}")
        except Exception as e:
            print(f"Error opening application: {e}")
    
    def _open_website(self, url: str, browser: str = "default"):
        """Open a website in specified browser"""
        try:
            if not url.startswith(('http://', 'https://')):
                url = f"https://{url}"
            
            if browser.lower() == "brave":
                brave_path = r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"
                if os.path.exists(brave_path):
                    subprocess.Popen([brave_path, url])
                else:
                    os.system(f"start {url}")
            else:
                os.system(f"start {url}")
            
            print(f"Opening website: {url}")
        except Exception as e:
            print(f"Error opening website: {e}")



def main():
    """
    Main function - redirects to Smart Voice System for unified interface
    """
    print("=== Automated Voice Command System ===")
    print("Note: This system is now integrated into the Smart Voice System.")
    print("Please use smart_voice_system.py for the complete interface.")
    print("\nStarting Smart Voice System...")
    
    # Import and run the smart system
    try:
        from smart_voice_system import main as smart_main
        smart_main()
    except ImportError:
        print("Error: Could not import smart_voice_system.py")
        print("Please run: python scripts/smart_voice_system.py")


if __name__ == "__main__":
    main()
