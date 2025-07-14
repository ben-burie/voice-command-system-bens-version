import os
import json
import torch
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Callable, Optional
import subprocess

from automated_command_system import AutomatedCommandSystem, EnhancedConformerVoiceSystem
from adaptive_incremental_trainer import AdaptiveIncrementalTrainer, SmartRetrainingScheduler

class SmartVoiceCommandSystem:
    
    def __init__(self, 
                 base_data_dir: str = "data_barkAI_large",
                 model_path: str = "models/conformer_best_model.pth",
                 config_path: str = "config/smart_system_config.json"):
        
        self.base_data_dir = Path(base_data_dir)
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        
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
                    "confidence_threshold": 0.7,
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
            self.voice_system = EnhancedConformerVoiceSystem(
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
    
    def start_system(self):
        """Start the complete smart voice system"""
        try:
            print("\n Starting Smart Voice Command System...")
            
            # Initialize voice system if not exists
            if not self.voice_system:
                self.voice_system = EnhancedConformerVoiceSystem(
                    str(self.model_path),
                    action_registry=getattr(self.automated_system, 'action_registry', {})
                )
            
            self.is_running = True
            self._start_voice_recognition()
            
            print(" Smart Voice System is running!")
            print(" Listening for voice commands...")
            print(" You can add new commands while the system is running")
            print(" Press Ctrl+C to stop")
            
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
        """Interactive command addition with smart training"""
        print("\n Smart Interactive Command Addition")
        
        command_name = input("Enter command name (e.g., 'Open_Discord'): ").strip()
        if not command_name:
            print(" Command name cannot be empty")
            return
        
        print("\nAction types:")
        print("1. Open application")
        print("2. Open website") 
        print("3. System command")
        print("4. Custom action")
        
        action_type = input("Select action type (1-4): ").strip()
        action_function = None
        action_info = None  # Initialize to None
        
        if action_type == "1":
            app_path = input("Enter application path/command: ").strip()
            action_function = lambda: self._execute_app(app_path)
            action_info = {"type": "app", "path": app_path}  # Create the dictionary
            
        elif action_type == "2":
            url = input("Enter website URL: ").strip()
            browser = input("Browser (brave/chrome/default): ").strip() or "default"
            action_function = lambda: self._execute_website(url, browser)
            action_info = {"type": "website", "url": url, "browser": browser} # Create the dictionary
            
        elif action_type == "3":
            cmd = input("Enter system command: ").strip()
            action_function = lambda: self._execute_system_command(cmd)
            
        elif action_type == "4":
            print("Custom action - implement in code")
            action_function = lambda: print(f"Custom action: {command_name}")
        
        else:
            print(" Invalid action type")
            return
        
        # Get number of samples
        num_samples = int(input("Number of audio samples (default 8): ") or "8")
        
        # Add command with smart training
        success = self.add_command_smart(command_name, action_info, action_function, num_samples)
        
        if success:
            print(f" Command '{command_name}' added successfully!")
            print(" Smart training will be handled automatically")
    
    def _execute_app(self, app_path: str):
        """Execute application"""
        try:
            subprocess.Popen(app_path, shell=True)
            print(f" Opened: {app_path}")
        except Exception as e:
            print(f" Error opening app: {e}")
    
    def _execute_website(self, url: str, browser: str):
        """Execute website opening"""
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
            
            print(f" Opened: {url}")
        except Exception as e:
            print(f" Error opening website: {e}")
    
    def _execute_system_command(self, cmd: str):
        """Execute system command"""
        try:
            subprocess.run(cmd, shell=True)
            print(f" Executed: {cmd}")
        except Exception as e:
            print(f" Error executing command: {e}")
    
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
        
        # Recent commands
        if all_commands:
            print(f"\n Available Commands:")
            for cmd in all_commands[-5:]:  # Show last 5
                print(f"   {cmd}")
            if len(all_commands) > 5:
                print(f"  ... and {len(all_commands) - 5} more")


def demo_actions():
    """Demo action functions"""
    
    def open_discord():
        try:
            discord_paths = [
                os.path.expanduser("~\\AppData\\Local\\Discord\\Update.exe --processStart Discord.exe"),
                "discord"
            ]
            for path in discord_paths:
                try:
                    subprocess.Popen(path, shell=True)
                    print(" Discord opened!")
                    return
                except:
                    continue
            print(" Discord not found")
        except Exception as e:
            print(f" Error opening Discord: {e}")
    
    def open_calculator():
        try:
            subprocess.Popen("calc", shell=True)
            print(" Calculator opened!")
        except Exception as e:
            print(f" Error opening calculator: {e}")
    
    def open_notepad():
        try:
            subprocess.Popen("notepad", shell=True)
            print(" Notepad opened!")
        except Exception as e:
            print(f" Error opening notepad: {e}")
    
    return {
        "Open_Discord": open_discord,
        "Open_Calculator": open_calculator,
        "Open_Notepad": open_notepad
    }


def main():
    """Main function for the Smart Voice Command System"""
    print(" Smart Voice Command System")
    print("=" * 50)
    
    # Initialize system
    smart_system = SmartVoiceCommandSystem()
    
    # Demo actions
    demo_funcs = demo_actions()
    
    try:
        while True:
            print("\n" + "=" * 50)
            print("1.  Add command (Smart)")
            print("2.  Start voice system")
            print("3.  Stop voice system")
            print("4.  Show system status")
            print("5.  Add demo commands")
            print("6.  Force retrain model")
            print("7.  Finetune model (pending commands)")
            print("8.  Exit")
            
            choice = input("\nSelect option (1-8): ").strip()
            
            if choice == "1":
                smart_system.add_command_interactive_smart()
                
            elif choice == "2":
                smart_system.start_system()
                try:
                    while smart_system.is_running:
                        time.sleep(1)
                except KeyboardInterrupt:
                    smart_system.stop_system()
                    
            elif choice == "3":
                smart_system.stop_system()
                
            elif choice == "4":
                smart_system.show_system_status()
                
            elif choice == "5":
                print(" Adding demo commands...")
                for cmd_name, cmd_func in demo_funcs.items():
                    print(f"Adding {cmd_name}...")
                    smart_system.add_command_smart(cmd_name, cmd_func, num_samples=6)
                    
            elif choice == "6":
                print(" Force retraining...")
                smart_system.automated_system.retrain_model()
                smart_system._reload_voice_system()
                
            elif choice == "7":
                if smart_system.pending_commands:
                    print(f" Finetuning model with pending commands: {smart_system.pending_commands}")
                    success = smart_system._finetune_model()
                    if success:
                        print(" Model finetuning completed successfully!")
                        smart_system.pending_commands.clear()
                        smart_system.config["system_stats"]["last_training"] = datetime.now().isoformat()
                        smart_system._save_config(smart_system.config)
                        smart_system._reload_voice_system()
                    else:
                        print(" Model finetuning failed!")
                else:
                    print(" No pending commands to finetune with.")
                
            elif choice == "8":
                smart_system.stop_system()
                print(" Goodbye!")
                break
                
            else:
                print(" Invalid option")
                
    except KeyboardInterrupt:
        smart_system.stop_system()
        print("\n System stopped by user")


if __name__ == "__main__":
    main()
