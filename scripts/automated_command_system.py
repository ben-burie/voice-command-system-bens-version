import os
import json
import torch
import torchaudio
import numpy as np
import sounddevice as sd
import threading
import time
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
            f"Please {command}",
            f"Can you {command}",
            f"I want to {command}",
            f"{command} now",
            f"{command}!",
            f"{command}."
        ])
        
        # Add speed variations through text manipulation
        slow_version = command.replace(" ", "  ")
        variations.append(slow_version)
        
        return variations
    
    def _apply_audio_augmentations(self, audio: np.ndarray, sample_rate: int) -> List[np.ndarray]:
        """Apply audio augmentations for robustness"""
        augmented_versions = [audio]  # Include original
        
        try:
            # Advanced augmentations with librosa
            slow_audio = librosa.effects.time_stretch(audio, rate=random.uniform(0.8, 0.95))
            fast_audio = librosa.effects.time_stretch(audio, rate=random.uniform(1.05, 1.3))
            augmented_versions.extend([slow_audio, fast_audio])
            
            # Pitch variations
            low_pitch = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=-1)
            high_pitch = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=1)
            augmented_versions.extend([low_pitch, high_pitch])
            
            # Basic augmentations (always available)
            # Volume variations
            quiet_audio = audio * 0.7
            loud_audio = np.clip(audio * 1.3, -1.0, 1.0)
            augmented_versions.extend([quiet_audio, loud_audio])
            
            # Add slight noise for robustness
            noise_factor = 0.003
            noisy_audio = audio + noise_factor * np.random.randn(len(audio))
            augmented_versions.append(noisy_audio)
            
        except Exception as e:
            print(f"Audio augmentation error: {e}")
        
        return augmented_versions
    
    def generate_bark_audio_samples(self, command_name: str, num_samples: int = None) -> List[str]:
        """
        Generate audio samples using Bark AI
        """
        if not BARK_AVAILABLE:
            print("Bark AI not available, falling back to manual recording")
            return self.record_audio_samples_manual(command_name, num_samples or 10)
        
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
                
                speaker_samples = 0
                max_speaker_samples = samples_per_speaker + (5 if speaker_idx < (num_samples % len(self.speakers)) else 0)
                
                while speaker_samples < max_speaker_samples and sample_count < num_samples:
                    try:
                        # Select random text variation
                        text_variant = random.choice(text_variations)
                        
                        # Generate audio with Bark
                        audio_array = generate_audio(text_variant, history_prompt=speaker)
                        
                        # Apply augmentations for robustness
                        augmented_audios = self._apply_audio_augmentations(audio_array, SAMPLE_RATE)
                        
                        # Save augmented versions
                        for aug_idx, final_audio in enumerate(augmented_audios):
                            if sample_count >= num_samples:
                                break
                            
                            # Normalize audio
                            final_audio = np.array(final_audio, dtype=np.float32)
                            if np.max(np.abs(final_audio)) > 0:
                                final_audio = final_audio / np.max(np.abs(final_audio)) * 0.9
                            
                            # Save file
                            filename = f"{command_name}_speaker{speaker_idx}_var{sample_count:03d}.wav"
                            filepath = command_dir / filename
                            
                            write_wav(str(filepath), SAMPLE_RATE, final_audio)
                            generated_files.append(str(filepath))
                            
                            sample_count += 1
                            speaker_samples += 1
                            pbar.update(1)
                            
                            if sample_count >= num_samples:
                                break
                    
                    except Exception as e:
                        print(f"Warning: Error generating sample {sample_count}: {e}")
                        continue
        
        print(f"Generated {len(generated_files)} audio files using Bark AI")
        return generated_files
    
    def record_audio_samples_manual(self, command_name: str, num_samples: int = 10) -> List[str]:
        """
        Manually record audio samples (fallback method)
        """
        print(f"\nRecording {num_samples} samples for command: '{command_name}'")
        print("Press ENTER when ready to start recording each sample...")
        
        # Create command directory
        command_dir = self.base_data_dir / command_name
        command_dir.mkdir(exist_ok=True)
        
        recorded_files = []
        
        for i in range(num_samples):
            input(f"\nSample {i+1}/{num_samples} - Press ENTER to start recording...")
            
            print(f"Recording sample {i+1}... Speak now!")
            
            # Record audio
            audio_data = sd.rec(
                int(self.recording_duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32
            )
            sd.wait()  # Wait for recording to complete
            
            print("Recording complete!")
            
            # Save audio file
            filename = f"{command_name}_sample_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            filepath = command_dir / filename
            
            # Convert to tensor and save
            audio_tensor = torch.from_numpy(audio_data.flatten())
            torchaudio.save(str(filepath), audio_tensor.unsqueeze(0), self.sample_rate)
            
            recorded_files.append(str(filepath))
            print(f"Saved: {filename}")
        
        return recorded_files
    
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
            
            # Generate or record audio samples
            if use_bark and BARK_AVAILABLE:
                print(" Using Bark AI for audio generation...")
                generated_files = self.generate_bark_audio_samples(command_name, num_samples)
                generation_method = "bark_ai"
            else:
                print(" Using manual recording...")
                generated_files = self.record_audio_samples_manual(command_name, num_samples or 10)
                generation_method = "manual_recording"
            
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
        Retrain the conformer model with all available commands
        """
        try:
            print("\n=== Retraining Conformer Model ===")
            
            # Import training components
            from trainConformer import BarkVoiceCommandDataset, train_model
            from torch.utils.data import DataLoader
            
            # Create datasets
            print("Creating training datasets...")
            train_dataset = BarkVoiceCommandDataset(str(self.base_data_dir), mode='train')
            val_dataset = BarkVoiceCommandDataset(str(self.base_data_dir), mode='val')
            
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
            
            # Initialize model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            num_classes = len(train_dataset.unique_labels)
            
            print(f"Training with {num_classes} classes: {train_dataset.unique_labels}")
            
            # Load existing model if available, otherwise create new
            model = ConformerModel(num_classes=num_classes).to(device)
            
            if self.model_path.exists():
                try:
                    checkpoint = torch.load(self.model_path, map_location=device)
                    # Only load if the number of classes matches
                    if model.classifier.out_features == checkpoint['classifier.weight'].shape[0]:
                        model.load_state_dict(checkpoint)
                        print(" Loaded existing model for fine-tuning")
                    else:
                        print(" Model architecture changed, training from scratch")
                except:
                    print(" Could not load existing model, training from scratch")
            
            # Train model with fewer epochs for fine-tuning
            epochs = 10 if self.model_path.exists() else 20
            print(f"Training for {epochs} epochs...")
            
            history = train_model(model, train_loader, val_loader, device, num_epochs=epochs)
            
            # Update configuration
            self.command_config["last_trained"] = datetime.now().isoformat()
            self.command_config["model_version"] += 1
            self._save_command_config(self.command_config)
            
            self.commands_since_retrain = 0
            
            print(" Model retrained successfully!")
            return True
            
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
            self.voice_system = EnhancedConformerVoiceSystem(
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
        print("3. Custom action")
        
        action_type = input("Select action type (1-3): ").strip()
        
        action_function = None
        
        if action_type == "1":
            app_path = input("Enter application path or command: ").strip()
            action_function = lambda: self._open_application(app_path)
            
        elif action_type == "2":
            url = input("Enter website URL: ").strip()
            browser = input("Enter browser (brave/chrome/default): ").strip() or "default"
            action_function = lambda: self._open_website(url, browser)
            
        elif action_type == "3":
            print("Custom action - you'll need to implement this in code")
            action_function = lambda: print(f"Executing custom action for {command_name}")
        
        else:
            print(" Invalid action type")
            return
        
        # Add the command
        num_samples = int(input("Number of audio samples to record (default 10): ") or "10")
        
        success = self.add_new_command(command_name, action_function, num_samples)
        
        if success:
            print(f" Command '{command_name}' added successfully!")
            
            # Ask if user wants to reload the voice system
            reload = input("Reload voice system now? (y/n): ").strip().lower()
            if reload == 'y':
                self.reload_voice_system()
        
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


class EnhancedConformerVoiceSystem(ConformerVoiceCommandSystem):
    """
    Enhanced voice system that supports dynamic command registration
    """
    
    def __init__(self, model_path, action_registry=None, **kwargs):
        super().__init__(model_path, **kwargs)
        
        # Merge with existing actions
        if action_registry:
            self.command_actions.update(action_registry)
        
        print(f"Enhanced system loaded with {len(self.command_actions)} actions")


def demo_discord_action():
    """Example action for opening Discord"""
    try:
        # Try to find Discord executable
        discord_paths = [
            os.path.expanduser("~\\AppData\\Local\\Discord\\Update.exe --processStart Discord.exe"),
            "discord",  # If Discord is in PATH
        ]
        
        for path in discord_paths:
            try:
                subprocess.Popen(path, shell=True)
                print("Opening Discord...")
                return
            except:
                continue
        
        print("Discord not found, please install Discord or update the path")
    except Exception as e:
        print(f"Error opening Discord: {e}")


def main():
    """
    Main function demonstrating the automated command system
    """
    print("=== Automated Voice Command System ===")
    
    # Initialize the system
    auto_system = AutomatedCommandSystem()
    
    while True:
        print("\n" + "="*50)
        print("1. Add new command interactively")
        print("2. Add 'Open Discord' command (demo)")
        print("3. Retrain model")
        print("4. Start voice command system")
        print("5. Show current commands")
        print("6. Exit")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == "1":
            auto_system.add_command_interactive()
            
        elif choice == "2":
            # Demo: Add Discord command
            print("\nAdding 'Open Discord' command...")
            success = auto_system.add_new_command(
                "Open_Discord", 
                demo_discord_action, 
                num_samples=8,
                auto_train=False
            )
            if success:
                print(" Discord command added! You can now retrain the model.")
            
        elif choice == "3":
            auto_system.retrain_model()
            
        elif choice == "4":
            print("Starting voice command system...")
            print("Say 'stop' or press 'q' to quit")
            auto_system.start_voice_system()
            
        elif choice == "5":
            commands = auto_system.command_config.get("commands", {})
            print(f"\nCurrent commands ({len(commands)}):")
            for cmd, info in commands.items():
                print(f"  - {cmd} (samples: {info.get('sample_count', 0)})")
            
        elif choice == "6":
            print("Goodbye!")
            break
            
        else:
            print("Invalid option, please try again.")


if __name__ == "__main__":
    main()

