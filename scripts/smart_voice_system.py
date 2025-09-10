import os
import json
import torch
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Callable, Optional
import subprocess

from automated_command_system import AutomatedCommandSystem
from voiceCommandConformer import ConformerVoiceCommandSystem
from adaptive_incremental_trainer import AdaptiveIncrementalTrainer, SmartRetrainingScheduler

class SmartVoiceCommandSystem:
    
    def __init__(self, 
                 base_data_dir: str = "data_barkAI_large",
                 model_path: str = "models\saved/conformer_best_model.pth",
                 config_path: str = "config/smart_system_config.json"):
        
        self.base_data_dir = Path(base_data_dir)
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)model_path: str = "models\saved/conformer_best_model.pth"
        
        self.automated_system = AutomatedCommandSystem(
            base_data_dir=str(self.base_data_dir),
            model_path=str(self.model_path),
            config_path=str(self.config_path.parent / "command_config.json")
        )
        
        self.incremental_trainer = AdaptiveIncrementalTrainer(
            model_path=str(self.model_path),
            base_data_dir=str(self.base_data_dir),
            config_path=str(self.config_path.parent / "adaptive_incremental_config.json")
        )
        
        self.scheduler = SmartRetrainingScheduler(
            config_path=str(self.config_path.parent / "scheduler_config.json")
        )
        
        self.voice_system = None
        self.is_running = False
        self.is_paused = False
        self.pending_commands = []
        
        self.config = self._load_config()
        
        print("Smart Voice Command System initialized")
        print(f"Data directory: {self.base_data_dir}")
        print(f"Model path: {self.model_path}")
    
    def _load_config(self) -> Dict:
        """Load system configuration"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            config = {
                "voice_recognition": {
                    "confidence_threshold": 0.75,
                    "continuous_listening": True,
                    "wake_word": None
                },
                "system_stats": {
                    "commands_added": 0,
                    "last_training": None,
                    "total_recognitions": 0
                }
            }
            self._save_config(config)
            return config
    
    def _save_config(self, config: Dict):
        """Save system configuration"""
        self.config_path.parent.mkdir(exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def add_command_smart(self, command_name: str, action_info: Dict, action_function: Callable, 
                         num_samples: int = 8) -> bool:
        """
        Add a new command and prompt for model finetuning
        """
        try:
            print(f"\n Smart Command Addition: {command_name}")
            
            # Add command using automated system
            success = self.automated_system.add_new_command(
                command_name=command_name,
                action_info=action_info,
                action_function=action_function,
                num_samples=num_samples,
                auto_train=False
            )
            
            if not success:
                return False
            
            # Update stats
            self.config["system_stats"]["commands_added"] += 1
            self._save_config(self.config)
            
            # Add to pending commands
            self.pending_commands.append(command_name)
            
            print(f" Command '{command_name}' added successfully!")
            print(f" Voice data generated for '{command_name}'")
            
            # Prompt user for finetuning
            finetune_choice = input("\nDo you want to finetune the model? (y/n): ").strip().lower()
            
            if finetune_choice == 'y' or finetune_choice == 'yes':
                print(" Starting model finetuning...")
                success = self._finetune_model()
                if success:
                    print(" Model finetuning completed successfully!")
                    self.pending_commands.clear()
                    self.config["system_stats"]["last_training"] = datetime.now().isoformat()
                    self._save_config(self.config)
                    # Reload voice system
                    self._reload_voice_system()
                else:
                    print(" Model finetuning failed!")
            else:
                print(" Model finetuning skipped. You can finetune later from the menu.")
            
            return True
            
        except Exception as e:
            print(f" Error in smart command addition: {e}")
            return False
    
    def _get_all_commands(self) -> List[str]:
        """Get list of all current commands"""
        commands = []
        for folder in self.base_data_dir.iterdir():
            if folder.is_dir():
                commands.append(folder.name)
        return commands
    
    def _finetune_model(self) -> bool:
        """
        Finetune the conformer model with pending commands
        """
        try:
            print(" Starting model finetuning...")
            
            # Use incremental training for finetuning
            success = self.incremental_trainer.incremental_train(
                new_command_dirs=self.pending_commands.copy(),
                epochs=5
            )
            
            if success:
                print(" Model finetuning completed successfully!")
                return True
            else:
                print(" Model finetuning failed!")
                return False
                
        except Exception as e:
            print(f" Error in model finetuning: {e}")
            return False
    
    
    def _reload_voice_system(self):
        """Reload the voice recognition system"""
        try:
            if self.voice_system and hasattr(self.voice_system, 'is_listening'):
                self.voice_system.stop_listening()
                time.sleep(1)
            
            # Create new voice system
            self.voice_system = ConformerVoiceCommandSystem(
                str(self.model_path),
                action_registry=getattr(self.automated_system, 'action_registry', {})
            )
            
            print(" Voice system reloaded with updated model")
            
            # Restart if it was running
            if self.is_running:
                self._start_voice_recognition()
            
        except Exception as e:
            print(f" Error reloading voice system: {e}")
    
    def _start_voice_recognition(self):
        """Start voice recognition in background"""
        def voice_worker():
            try:
                if self.voice_system:
                    self.voice_system.start()
            except Exception as e:
                print(f" Error in voice recognition: {e}")
        
        thread = threading.Thread(target=voice_worker, daemon=True)
        thread.start()
    
    def pause_system(self):
        """Pause voice recognition for presentations"""
        if self.is_running and not self.is_paused:
            self.is_paused = True
            # Set pause flag on voice system
            if self.voice_system:
                self.voice_system.is_paused = True
            print("\nVoice system PAUSED - Safe for presentations")
            print("   Voice commands will not be processed")
            print("   Press 'r' to resume listening")
        else:
            print("\n Voice system is not running or already paused")
    
    def resume_system(self):
        """Resume voice recognition after pause"""
        if self.is_running and self.is_paused:
            self.is_paused = False
            # Clear pause flag on voice system
            if self.voice_system:
                self.voice_system.is_paused = False
            print("\nVoice system RESUMED - Listening for commands")
            print("   Press 'p' to pause for presentations")
        else:
            print("\n Voice system is not paused or not running")
    
    def toggle_pause(self):
        """Toggle pause/resume state"""
        if self.is_paused:
            self.resume_system()
        else:
            self.pause_system()
    
    def start_system(self):
        """Start the complete smart voice system"""
        try:
            print("\n Starting Smart Voice Command System...")
            
            # Initialize voice system if not exists
            if not self.voice_system:
                self.voice_system = ConformerVoiceCommandSystem(
                    str(self.model_path),
                    action_registry=getattr(self.automated_system, 'action_registry', {})
                )
            
            self.is_running = True
            self.is_paused = False
            self._start_voice_recognition()
            
            print(" Smart Voice System is running!")
            print(" Listening for voice commands...")
            print(" You can add new commands while the system is running")
            print("\nPRESENTATION CONTROLS:")
            print("   Press 'p' to PAUSE voice recognition (safe for presentations)")
            print("   Press 'r' to RESUME voice recognition")
            print("   Press 'Ctrl+C' to stop completely")
            
        except Exception as e:
            print(f" Error starting system: {e}")
    
    def stop_system(self):
        """Stop the smart voice system"""
        try:
            print("\n Stopping Smart Voice System...")
            self.is_running = False
            
            if self.voice_system and hasattr(self.voice_system, 'stop_listening'):
                self.voice_system.stop_listening()
            
            print(" System stopped")
            
        except Exception as e:
            print(f" Error stopping system: {e}")
    
    def add_command_interactive_smart(self):
        """Interactive command addition with smart training - delegates to automated system"""
        # Use the automated system's interactive function but with smart training
        return self.automated_system.add_command_interactive()
    
    def show_system_status(self):
        """Show current system status"""
        print("\n Smart Voice System Status")
        print("=" * 40)
        
        # Commands
        all_commands = self._get_all_commands()
        print(f" Total Commands: {len(all_commands)}")
        print(f" Pending Training: {len(self.pending_commands)}")
        
        # System stats
        stats = self.config["system_stats"]
        print(f" Commands Added: {stats['commands_added']}")
        print(f" Last Training: {stats.get('last_training', 'Never')}")
        print(f" System Running: {'Yes' if self.is_running else 'No'}")
        if self.is_running:
            status_icon = "PAUSED" if self.is_paused else "LISTENING"
            print(f" Voice Status: {status_icon}")
        
        # Recent commands
        if all_commands:
            print(f"\n Available Commands:")
            for cmd in all_commands[-5:]:  # Show last 5
                print(f"   {cmd}")
            if len(all_commands) > 5:
                print(f"  ... and {len(all_commands) - 5} more")

def main():
    """Main function for the Smart Voice Command System"""
    print("Smart Voice Command System")
    print("=" * 50)
    
    # Initialize system
    smart_system = SmartVoiceCommandSystem()
    
    try:
        while True:
            print("\n" + "=" * 50)
            print("MAIN MENU")
            print("=" * 50)
            print("1.  Add command (Smart with incremental training)")
            print("2.  Start voice system")
            print("3.  Stop voice system")
            print("4.  Show system status & commands")
            print("5.  Force retrain model (full retraining)")
            print("6.  Finetune model (pending commands only)")
            print("7.  Exit")
            
            choice = input("\nSelect option (1-7): ").strip()
            
            if choice == "1":
                # Smart command addition with incremental training
                smart_system.add_command_interactive_smart()
                
            elif choice == "2":
                print("\nStarting voice system...")
                smart_system.start_system()
                try:
                    import keyboard
                    while smart_system.is_running:
                        # Check for keyboard input
                        if keyboard.is_pressed('p'):
                            if not smart_system.is_paused:
                                smart_system.pause_system()
                                time.sleep(0.5)  # Prevent multiple triggers
                        elif keyboard.is_pressed('r'):
                            if smart_system.is_paused:
                                smart_system.resume_system()
                                time.sleep(0.5)  # Prevent multiple triggers
                        time.sleep(0.1)
                except ImportError:
                    print("Keyboard module not available. Using basic mode without pause/resume controls.")
                    try:
                        while smart_system.is_running:
                            time.sleep(1)
                    except KeyboardInterrupt:
                        smart_system.stop_system()
                except KeyboardInterrupt:
                    smart_system.stop_system()
                    
            elif choice == "3":
                print("\nStopping voice system...")
                smart_system.stop_system()
                
            elif choice == "4":
                print("\nSystem Status")
                smart_system.show_system_status()
                
                # Also show detailed command list
                commands = smart_system.automated_system.command_config.get("commands", {})
                if commands:
                    print(f"\nDetailed Command List ({len(commands)} total):")
                    for cmd, info in commands.items():
                        method = info.get('generation_method', 'unknown')
                        samples = info.get('sample_count', 0)
                        date = info.get('added_date', 'unknown')[:10] if info.get('added_date') else 'unknown'
                        action_type = info.get('action', {}).get('type', 'unknown')
                        print(f"   • {cmd}")
                        print(f"     └─ Type: {action_type} | Samples: {samples} | Method: {method} | Added: {date}")
                else:
                    print("\nNo commands configured yet.")
                    
            elif choice == "5":
                print("\n Force retraining (full model retraining)...")
                print("This will retrain the entire model from scratch.")
                confirm = input("Continue? (y/n): ").strip().lower()
                if confirm == 'y':
                    success = smart_system.automated_system.retrain_model()
                    if success:
                        smart_system._reload_voice_system()
                        print("Full retraining completed!")
                    else:
                        print("Full retraining failed!")
                else:
                    print("Retraining cancelled.")
                
            elif choice == "6":
                print("\nIncremental finetuning...")
                if smart_system.pending_commands:
                    print(f"Finetuning with pending commands: {smart_system.pending_commands}")
                    success = smart_system._finetune_model()
                    if success:
                        print("Incremental finetuning completed successfully!")
                        smart_system.pending_commands.clear()
                        smart_system.config["system_stats"]["last_training"] = datetime.now().isoformat()
                        smart_system._save_config(smart_system.config)
                        smart_system._reload_voice_system()
                    else:
                        print("Incremental finetuning failed!")
                else:
                    print("No pending commands to finetune with.")
                    print("Add commands using option 1 to create pending commands for finetuning.")
                
            elif choice == "7":
                print("\nShutting down...")
                smart_system.stop_system()
                print("Goodbye!")
                break
                
            else:
                print("Invalid option. Please select 1-7.")
                
    except KeyboardInterrupt:
        smart_system.stop_system()
        print("\nSystem stopped by user")


if __name__ == "__main__":
    main()
