import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
import json
import sys
import os
from pathlib import Path
from datetime import datetime

# Add the scripts directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

try:
    from smart_voice_system import SmartVoiceCommandSystem
except ImportError as e:
    print(f"Error importing smart_voice_system: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)

class VoiceSystemGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Voice Command System - Demo UI")
        self.root.geometry("900x1000")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize the voice system
        self.voice_system = None
        self.is_initializing = False
        
        # Create the UI
        self.create_widgets()
        
        # Start status update thread
        self.update_thread_running = True
        self.update_thread = threading.Thread(target=self.update_status_loop, daemon=True)
        self.update_thread.start()
        
        # Initialize voice system in background
        self.initialize_voice_system()
    
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Smart Speech", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Status Frame
        status_frame = ttk.LabelFrame(main_frame, text="System Status", padding="10")
        status_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        status_frame.columnconfigure(1, weight=1)
        
        # Status indicators
        ttk.Label(status_frame, text="System State:").grid(row=0, column=0, sticky=tk.W)
        self.status_label = ttk.Label(status_frame, text="Initializing...", foreground="orange")
        self.status_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        ttk.Label(status_frame, text="Total Commands:").grid(row=1, column=0, sticky=tk.W)
        self.commands_count_label = ttk.Label(status_frame, text="0")
        self.commands_count_label.grid(row=1, column=1, sticky=tk.W, padx=(10, 0))
        
        ttk.Label(status_frame, text="Pending Training:").grid(row=2, column=0, sticky=tk.W)
        self.pending_label = ttk.Label(status_frame, text="0")
        self.pending_label.grid(row=2, column=1, sticky=tk.W, padx=(10, 0))
        
        # Control Frame
        control_frame = ttk.LabelFrame(main_frame, text="System Controls", padding="10")
        control_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        button_frame.columnconfigure((0, 1, 2, 3), weight=1)
        
        self.start_btn = ttk.Button(button_frame, text="Start Voice System", 
                                   command=self.start_voice_system, state="disabled")
        self.start_btn.grid(row=0, column=0, padx=(0, 5), sticky=(tk.W, tk.E))
        
        self.stop_btn = ttk.Button(button_frame, text="Stop Voice System", 
                                  command=self.stop_voice_system, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))
        
        self.pause_btn = ttk.Button(button_frame, text="Pause (Demo Mode)", 
                                   command=self.pause_voice_system, state="disabled")
        self.pause_btn.grid(row=0, column=2, padx=5, sticky=(tk.W, tk.E))
        
        self.resume_btn = ttk.Button(button_frame, text="Resume Listening", 
                                    command=self.resume_voice_system, state="disabled")
        self.resume_btn.grid(row=0, column=3, padx=(5, 0), sticky=(tk.W, tk.E))
        
        # Add Command Frame
        add_frame = ttk.LabelFrame(main_frame, text="Add New Command", padding="10")
        add_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        add_frame.columnconfigure(1, weight=1)
        
        # Command name
        ttk.Label(add_frame, text="Command Name:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.command_name_entry = ttk.Entry(add_frame, width=30)
        self.command_name_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(0, 5))
        
        # Action type
        ttk.Label(add_frame, text="Action Type:").grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        self.action_type_var = tk.StringVar()
        self.action_type_combo = ttk.Combobox(add_frame, textvariable=self.action_type_var,
                                             values=["Open Application", "Open Website", "System Command", "Custom Action"],
                                             state="readonly")
        self.action_type_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(0, 5))
        self.action_type_combo.bind('<<ComboboxSelected>>', self.on_action_type_change)
        
        # Action details
        ttk.Label(add_frame, text="Details:").grid(row=2, column=0, sticky=tk.W, pady=(0, 5))
        self.action_details_entry = ttk.Entry(add_frame, width=30)
        self.action_details_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(0, 5))
        
        # Number of samples
        ttk.Label(add_frame, text="Samples:").grid(row=3, column=0, sticky=tk.W, pady=(0, 10))
        self.samples_var = tk.StringVar(value="8")
        samples_spinbox = ttk.Spinbox(add_frame, from_=1, to=50, textvariable=self.samples_var, width=10)
        samples_spinbox.grid(row=3, column=1, sticky=tk.W, padx=(10, 0), pady=(0, 10))
        
        # Audio Generation Progress
        ttk.Label(add_frame, text="Audio Progress:").grid(row=4, column=0, sticky=tk.W, pady=(0, 5))
        self.audio_progress = ttk.Progressbar(add_frame, mode='determinate')
        self.audio_progress.grid(row=4, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(0, 5))
        self.audio_progress_label = ttk.Label(add_frame, text="Ready")
        self.audio_progress_label.grid(row=5, column=1, sticky=tk.W, padx=(10, 0), pady=(0, 5))
        
        # Add button
        self.add_command_btn = ttk.Button(add_frame, text="Add Command", 
                                         command=self.add_command, state="disabled")
        self.add_command_btn.grid(row=6, column=0, columnspan=2, pady=(10, 0))
        
        # Fine-tuning Frame
        finetune_frame = ttk.LabelFrame(main_frame, text="Model Fine-tuning", padding="10")
        finetune_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        finetune_frame.columnconfigure(1, weight=1)
        
        # Fine-tuning progress
        ttk.Label(finetune_frame, text="Training Progress:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.finetune_progress = ttk.Progressbar(finetune_frame, mode='determinate')
        self.finetune_progress.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(0, 5))
        self.finetune_progress_label = ttk.Label(finetune_frame, text="Ready")
        self.finetune_progress_label.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=(0, 5))
        
        # Queue display
        ttk.Label(finetune_frame, text="Queue:").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        self.queue_text = tk.Text(finetune_frame, height=3, width=50, state='disabled')
        self.queue_text.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(5, 0))
        
        # Fine-tune button
        self.finetune_btn = ttk.Button(finetune_frame, text="Start Fine-tuning", 
                                      command=self.start_finetune, state="disabled")
        self.finetune_btn.grid(row=3, column=0, columnspan=2, pady=(10, 0))
        
        # Commands List Frame
        commands_frame = ttk.LabelFrame(main_frame, text="Available Commands", padding="10")
        commands_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        commands_frame.columnconfigure(0, weight=1)
        commands_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(5, weight=1)
        
        # Commands listbox with scrollbar
        commands_list_frame = ttk.Frame(commands_frame)
        commands_list_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        commands_list_frame.columnconfigure(0, weight=1)
        commands_list_frame.rowconfigure(0, weight=1)
        
        self.commands_listbox = tk.Listbox(commands_list_frame, height=6)
        self.commands_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        commands_scrollbar = ttk.Scrollbar(commands_list_frame, orient="vertical", command=self.commands_listbox.yview)
        commands_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.commands_listbox.configure(yscrollcommand=commands_scrollbar.set)
        
        # Activity Log Frame
        log_frame = ttk.LabelFrame(main_frame, text="Activity Log", padding="10")
        log_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(6, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, state='disabled')
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Clear log button
        ttk.Button(log_frame, text="Clear Log", command=self.clear_log).grid(row=1, column=0, pady=(5, 0))
        
        # Initial log message
        self.log_message("System initializing... Please wait.")
    
    def initialize_voice_system(self):
        """Initialize the voice system in a separate thread"""
        def init_worker():
            try:
                self.is_initializing = True
                self.log_message("Initializing Smart Voice Command System...")
                
                # Change to the correct directory
                original_dir = os.getcwd()
                project_dir = Path(__file__).parent.parent
                os.chdir(project_dir)
                
                self.voice_system = SmartVoiceCommandSystem()
                self.log_message("Voice system initialized successfully!")
                
                # Enable buttons
                self.root.after(0, self.enable_controls)
                
                os.chdir(original_dir)
                
            except Exception as e:
                self.log_message(f"Error initializing voice system: {e}")
                self.root.after(0, lambda: messagebox.showerror("Initialization Error", 
                                                               f"Failed to initialize voice system:\n{e}"))
            finally:
                self.is_initializing = False
        
        threading.Thread(target=init_worker, daemon=True).start()
    
    def enable_controls(self):
        """Enable control buttons after initialization"""
        self.start_btn.configure(state="normal")
        self.add_command_btn.configure(state="normal")
        self.finetune_btn.configure(state="normal")
        self.status_label.configure(text="Ready", foreground="green")
    
    def on_action_type_change(self, event=None):
        """Update placeholder text based on action type"""
        action_type = self.action_type_var.get()
        if action_type == "Open Application":
            self.action_details_entry.delete(0, tk.END)
            self.action_details_entry.insert(0, "e.g., notepad.exe or Discord")
        elif action_type == "Open Website":
            self.action_details_entry.delete(0, tk.END)
            self.action_details_entry.insert(0, "e.g., https://google.com")
        elif action_type == "System Command":
            self.action_details_entry.delete(0, tk.END)
            self.action_details_entry.insert(0, "e.g., shutdown /s /t 0")
        elif action_type == "Custom Action":
            self.action_details_entry.delete(0, tk.END)
            self.action_details_entry.insert(0, "Custom action description")
    
    def start_voice_system(self):
        """Start the voice recognition system"""
        if not self.voice_system:
            messagebox.showerror("Error", "Voice system not initialized")
            return
        
        def start_worker():
            try:
                self.log_message("Starting voice recognition system...")
                self.voice_system.start_system()
                self.root.after(0, self.on_voice_system_started)
            except Exception as e:
                self.log_message(f"Error starting voice system: {e}")
                self.root.after(0, lambda: messagebox.showerror("Start Error", f"Failed to start voice system:\n{e}"))
        
        threading.Thread(target=start_worker, daemon=True).start()
    
    def on_voice_system_started(self):
        """Update UI when voice system starts"""
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.pause_btn.configure(state="normal")
        self.status_label.configure(text="Listening", foreground="blue")
        self.log_message("Voice system started - Listening for commands")
    
    def stop_voice_system(self):
        """Stop the voice recognition system"""
        if not self.voice_system:
            return
        
        def stop_worker():
            try:
                self.log_message("Stopping voice recognition system...")
                self.voice_system.stop_system()
                self.root.after(0, self.on_voice_system_stopped)
            except Exception as e:
                self.log_message(f"Error stopping voice system: {e}")
        
        threading.Thread(target=stop_worker, daemon=True).start()
    
    def on_voice_system_stopped(self):
        """Update UI when voice system stops"""
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.pause_btn.configure(state="disabled")
        self.resume_btn.configure(state="disabled")
        self.status_label.configure(text="Stopped", foreground="red")
        self.log_message("Voice system stopped")
    
    def pause_voice_system(self):
        """Pause voice recognition for demo purposes"""
        if not self.voice_system:
            return
        
        try:
            self.voice_system.pause_system()
            self.pause_btn.configure(state="disabled")
            self.resume_btn.configure(state="normal")
            self.status_label.configure(text="Paused (Demo Mode)", foreground="orange")
            self.log_message("Voice system paused - Safe for demonstrations")
        except Exception as e:
            self.log_message(f"Error pausing voice system: {e}")
    
    def resume_voice_system(self):
        """Resume voice recognition after pause"""
        if not self.voice_system:
            return
        
        try:
            self.voice_system.resume_system()
            self.pause_btn.configure(state="normal")
            self.resume_btn.configure(state="disabled")
            self.status_label.configure(text="Listening", foreground="blue")
            self.log_message("Voice system resumed - Listening for commands")
        except Exception as e:
            self.log_message(f"Error resuming voice system: {e}")
    
    def add_command(self):
        """Add a new voice command"""
        if not self.voice_system:
            messagebox.showerror("Error", "Voice system not initialized")
            return
        
        command_name = self.command_name_entry.get().strip()
        action_type = self.action_type_var.get()
        action_details = self.action_details_entry.get().strip()
        
        if not command_name:
            messagebox.showerror("Error", "Please enter a command name")
            return
        
        if not action_type:
            messagebox.showerror("Error", "Please select an action type")
            return
        
        if not action_details and action_type != "Custom Action":
            messagebox.showerror("Error", "Please enter action details")
            return
        
        try:
            num_samples = int(self.samples_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid number of samples")
            return
        
        def add_worker():
            try:
                self.log_message(f"Adding command: {command_name}")
                
                # Create action info and function based on type
                if action_type == "Open Application":
                    action_info = {"type": "app", "path": action_details}
                    action_function = lambda: self.voice_system.automated_system._open_application(action_details)
                elif action_type == "Open Website":
                    action_info = {"type": "website", "url": action_details, "browser": "default"}
                    action_function = lambda: self.voice_system.automated_system._open_website(action_details)
                elif action_type == "System Command":
                    action_info = {"type": "system", "command": action_details}
                    action_function = lambda: self.voice_system.automated_system._execute_system_command(action_details)
                else:  # Custom Action
                    action_info = {"type": "custom"}
                    action_function = lambda: print(f"Executing custom action for {command_name}")
                
                # Show audio generation progress
                self.log_message(f"Generating audio samples for '{command_name}'...")
                self.root.after(0, lambda: self.update_audio_progress(0, f"Starting audio generation..."))
                
                # Simulate realistic progress during audio generation
                for i in range(0, num_samples):
                    progress = int((i / num_samples) * 100)
                    self.root.after(0, lambda p=progress, s=i+1: self.update_audio_progress(p, f"Generating sample {s}/{num_samples}"))
                    time.sleep(0.1)  # Small delay to show progress
                
                # Add the command (without auto fine-tuning)
                success = self.voice_system.automated_system.add_new_command(
                    command_name=command_name,
                    action_info=action_info,
                    action_function=action_function,
                    num_samples=num_samples,
                    auto_train=False
                )
                
                if success:
                    # Update stats
                    self.voice_system.config["system_stats"]["commands_added"] += 1
                    self.voice_system._save_config(self.voice_system.config)
                    
                    # Add to pending commands
                    self.voice_system.pending_commands.append(command_name)
                    
                    self.root.after(0, lambda: self.update_audio_progress(100, f"Audio generation complete!"))
                    self.log_message(f"Command '{command_name}' added successfully!")
                    self.log_message(f"Voice data generated for '{command_name}'")
                    
                    # Ask user if they want to fine-tune
                    self.root.after(0, lambda: self.ask_finetune_dialog(command_name))
                    self.root.after(0, self.on_command_added)
                else:
                    self.root.after(0, lambda: self.update_audio_progress(0, "Failed to generate audio"))
                    self.log_message(f"Failed to add command '{command_name}'")
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to add command '{command_name}'"))
                
            except Exception as e:
                self.log_message(f"Error adding command: {e}")
                self.root.after(0, lambda: messagebox.showerror("Add Command Error", f"Failed to add command:\n{e}"))
        
        threading.Thread(target=add_worker, daemon=True).start()
    
    def on_command_added(self):
        """Clear form after successful command addition"""
        self.command_name_entry.delete(0, tk.END)
        self.action_type_combo.set("")
        self.action_details_entry.delete(0, tk.END)
        self.samples_var.set("8")
        self.update_commands_list()
        self.update_queue_display()
        
        # Reset audio progress after a delay
        def reset_progress():
            time.sleep(2)
            self.root.after(0, lambda: self.update_audio_progress(0, "Ready"))
        threading.Thread(target=reset_progress, daemon=True).start()
    
    def update_commands_list(self):
        """Update the commands list display"""
        if not self.voice_system:
            return
        
        try:
            # Clear current list
            self.commands_listbox.delete(0, tk.END)
            
            # Get commands from the system
            commands = self.voice_system._get_all_commands()
            
            if commands:
                for cmd in commands:
                    self.commands_listbox.insert(tk.END, cmd)
            else:
                self.commands_listbox.insert(tk.END, "No commands available")
            
            # Update count
            self.commands_count_label.configure(text=str(len(commands)))
            
        except Exception as e:
            self.log_message(f"Error updating commands list: {e}")
    
    def update_status_loop(self):
        """Background thread to update status information"""
        while self.update_thread_running:
            try:
                if self.voice_system and not self.is_initializing:
                    # Update commands list
                    self.root.after(0, self.update_commands_list)
                    
                    # Update pending commands count
                    pending_count = len(getattr(self.voice_system, 'pending_commands', []))
                    self.root.after(0, lambda: self.pending_label.configure(text=str(pending_count)))
                    
                    # Update queue display
                    self.root.after(0, self.update_queue_display)
                
                time.sleep(2)  # Update every 2 seconds
                
            except Exception as e:
                print(f"Error in status update: {e}")
                time.sleep(5)
    
    def log_message(self, message):
        """Add a message to the activity log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        def update_log():
            self.log_text.configure(state='normal')
            self.log_text.insert(tk.END, log_entry)
            self.log_text.configure(state='disabled')
            self.log_text.see(tk.END)
        
        if threading.current_thread() == threading.main_thread():
            update_log()
        else:
            self.root.after(0, update_log)
    
    def update_audio_progress(self, progress, status_text):
        """Update the audio generation progress bar"""
        self.audio_progress['value'] = progress
        self.audio_progress_label.configure(text=status_text)
        self.root.update_idletasks()
    
    def update_finetune_progress(self, progress, status_text):
        """Update the fine-tuning progress bar"""
        self.finetune_progress['value'] = progress
        self.finetune_progress_label.configure(text=status_text)
        self.root.update_idletasks()
    
    def update_queue_display(self):
        """Update the fine-tuning queue display"""
        if not self.voice_system:
            return
        
        try:
            pending_commands = getattr(self.voice_system, 'pending_commands', [])
            
            self.queue_text.configure(state='normal')
            self.queue_text.delete(1.0, tk.END)
            
            if pending_commands:
                queue_text = f"Commands pending fine-tuning ({len(pending_commands)}):\n"
                for i, cmd in enumerate(pending_commands, 1):
                    queue_text += f"{i}. {cmd}\n"
            else:
                queue_text = "No commands pending fine-tuning"
            
            self.queue_text.insert(1.0, queue_text)
            self.queue_text.configure(state='disabled')
            
        except Exception as e:
            self.log_message(f"Error updating queue display: {e}")
    
    def show_progress_in_log(self, task_name, progress):
        """Show progress bar in activity log"""
        bar_length = 20
        filled_length = int(bar_length * progress // 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        progress_text = f"{task_name}: [{bar}] {progress}%"
        
        def update_progress():
            self.log_text.configure(state='normal')
            # Remove the last line if it's a progress update for the same task
            content = self.log_text.get(1.0, tk.END)
            lines = content.strip().split('\n')
            if lines and task_name in lines[-1] and '[' in lines[-1]:
                # Remove last line
                self.log_text.delete(f"{len(lines)}.0", tk.END)
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] {progress_text}\n"
            self.log_text.insert(tk.END, log_entry)
            self.log_text.configure(state='disabled')
            self.log_text.see(tk.END)
        
        if threading.current_thread() == threading.main_thread():
            update_progress()
        else:
            self.root.after(0, update_progress)
    
    def ask_finetune_dialog(self, command_name):
        """Ask user if they want to fine-tune the model"""
        result = messagebox.askyesno(
            "Fine-tune Model?",
            f"Command '{command_name}' has been added successfully!\n\n"
            f"Would you like to fine-tune the model now to improve recognition accuracy?\n\n"
            f"• Yes: Start fine-tuning immediately (recommended)\n"
            f"• No: Skip for now (you can fine-tune later)",
            icon='question'
        )
        
        if result:
            self.start_finetune()
        else:
            self.log_message("Fine-tuning skipped. You can fine-tune later from the menu.")
    
    def start_finetune(self):
        """Start the fine-tuning process"""
        if not self.voice_system:
            messagebox.showerror("Error", "Voice system not initialized")
            return
        
        pending_commands = getattr(self.voice_system, 'pending_commands', [])
        if not pending_commands:
            messagebox.showinfo("Info", "No commands pending fine-tuning")
            return
        
        def finetune_worker():
            try:
                self.log_message("Starting model fine-tuning...")
                self.root.after(0, lambda: self.update_finetune_progress(0, "Initializing fine-tuning..."))
                
                # Disable the fine-tune button during training
                self.root.after(0, lambda: self.finetune_btn.configure(state="disabled"))
                
                # Simulate realistic fine-tuning progress
                total_commands = len(pending_commands)
                
                for i, command in enumerate(pending_commands):
                    base_progress = int((i / total_commands) * 80)  # 80% for processing commands
                    
                    # Processing each command
                    self.root.after(0, lambda p=base_progress, c=command: 
                                   self.update_finetune_progress(p, f"Processing '{c}'..."))
                    time.sleep(0.5)
                    
                    # Training on command data
                    for j in range(5):  # Simulate training epochs
                        progress = base_progress + int((j / 5) * (80 / total_commands))
                        self.root.after(0, lambda p=progress, c=command, e=j+1: 
                                       self.update_finetune_progress(p, f"Training '{c}' - Epoch {e}/5"))
                        time.sleep(0.3)
                
                # Final model optimization
                self.root.after(0, lambda: self.update_finetune_progress(85, "Optimizing model..."))
                time.sleep(1)
                
                self.root.after(0, lambda: self.update_finetune_progress(95, "Saving model..."))
                time.sleep(0.5)
                
                # Call the actual fine-tuning method
                success = self.voice_system._finetune_model()
                
                if success:
                    self.root.after(0, lambda: self.update_finetune_progress(100, "Fine-tuning completed!"))
                    self.log_message("Model fine-tuning completed successfully!")
                    self.voice_system.pending_commands.clear()
                    self.voice_system.config["system_stats"]["last_training"] = datetime.now().isoformat()
                    self.voice_system._save_config(self.voice_system.config)
                    
                    # Update queue display
                    self.root.after(0, self.update_queue_display)
                    
                    # Reload voice system if it's running
                    if self.voice_system.is_running:
                        self.log_message("Reloading voice system with updated model...")
                        self.voice_system._reload_voice_system()
                        self.log_message("Voice system reloaded successfully!")
                else:
                    self.root.after(0, lambda: self.update_finetune_progress(0, "Fine-tuning failed!"))
                    self.log_message("Model fine-tuning failed!")
                
                # Re-enable the fine-tune button
                self.root.after(0, lambda: self.finetune_btn.configure(state="normal"))
                
                # Reset progress after delay
                def reset_progress():
                    time.sleep(3)
                    self.root.after(0, lambda: self.update_finetune_progress(0, "Ready"))
                threading.Thread(target=reset_progress, daemon=True).start()
                    
            except Exception as e:
                self.log_message(f"Error during fine-tuning: {e}")
                self.root.after(0, lambda: self.update_finetune_progress(0, "Error occurred"))
                self.root.after(0, lambda: self.finetune_btn.configure(state="normal"))
        
        threading.Thread(target=finetune_worker, daemon=True).start()
    
    def clear_log(self):
        """Clear the activity log"""
        self.log_text.configure(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state='disabled')
        self.log_message("Log cleared")
    
    def on_closing(self):
        """Handle window closing"""
        self.update_thread_running = False
        if self.voice_system:
            try:
                self.voice_system.stop_system()
            except:
                pass
        self.root.destroy()

def main():
    root = tk.Tk()
    app = VoiceSystemGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
