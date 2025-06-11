import torch
import torch.nn as nn
import sounddevice as sd
import numpy as np
import threading
import queue
import time
import torchaudio
import os
import math
from pathlib import Path
import keyboard
import subprocess
import win32com.client
# ------------------------ Conformer Model Components ------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

class FeedForwardModule(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForwardModule, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = residual + x
        x = self.layer_norm(x)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(0, 1)  # (batch, seq_len, d_model) -> (seq_len, batch, d_model)
        x, _ = self.multihead_attn(x, x, x)
        x = x.transpose(0, 1)  # (seq_len, batch, d_model) -> (batch, seq_len, d_model)
        x = self.dropout(x)
        x = residual + x
        return x

class ConvModule(nn.Module):
    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super(ConvModule, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, padding=padding, groups=d_model
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # (batch, seq_len, d_model) -> (batch, d_model, seq_len)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # (batch, d_model, seq_len) -> (batch, seq_len, d_model)
        x = residual + x
        return x

class ConformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, kernel_size, dropout=0.1):
        super(ConformerBlock, self).__init__()
        self.ff_module1 = FeedForwardModule(d_model, d_ff, dropout)
        self.self_attn_module = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.conv_module = ConvModule(d_model, kernel_size, dropout)
        self.ff_module2 = FeedForwardModule(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + 0.5 * self.ff_module1(x)
        x = self.self_attn_module(x)
        x = self.conv_module(x)
        x = x + 0.5 * self.ff_module2(x)
        x = self.layer_norm(x)
        return x

# ------------------------ Conformer Model ------------------------
class ConformerModel(nn.Module):
    def __init__(self, num_classes, d_model=144, d_ff=256, num_heads=4, 
                 num_layers=4, kernel_size=31, dropout=0.1):
        super(ConformerModel, self).__init__()
        
        self.d_model = d_model
        
        # Convolutional subsampling with additional pooling for longer sequences
        self.conv_subsampling = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU()
        )
        self.input_proj = nn.Linear(64, d_model)
        
        # Increase max_len for positional encoding to handle longer sequences
        self.pos_encoding = PositionalEncoding(d_model, max_len=5000)
        
        # Conformer blocks
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(d_model, d_ff, num_heads, kernel_size, dropout)
            for _ in range(num_layers)
        ])
        
        # Use a single Linear layer for classifier instead of Sequential
        self.classifier = nn.Linear(d_model, num_classes)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        # x shape: (batch, channels, freq, time) = (batch, 1, 64, 64)
        batch_size = x.size(0)
        
        # Apply convolutional subsampling
        x = self.conv_subsampling(x)  # (batch, 64, H, W)
        
        # Reshape for sequence processing
        x = x.permute(0, 2, 3, 1)  # (batch, H, W, 64)
        x = x.reshape(batch_size, -1, x.size(3))  # (batch, seq_len, feature_dim)
        
        # Project to model dimension - use input_proj to match saved model
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply Conformer blocks
        for block in self.conformer_blocks:
            x = block(x)
        
        # Global pooling
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x = self.global_pool(x).squeeze(-1)  # (batch, d_model)
        
        # Classification - use single classifier layer to match saved model
        x = self.classifier(x)
        
        return x

# ------------------------ Voice Command System ------------------------
class ConformerVoiceCommandSystem:
    def __init__(self, model_path, sample_rate=16000, window_duration=4):
        self.debug_mode = True
        self.sample_rate = sample_rate
        self.window_duration = window_duration
        self.window_samples = int(sample_rate * window_duration)
        
        print("Initializing Conformer model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load the command labels from the data directory
        self.commands = self._load_command_labels("data_barkAI")
        num_classes = len(self.commands)
        print(f"Detected {num_classes} command classes: {self.commands}")
        
        self.model = ConformerModel(num_classes=num_classes).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        print("Conformer model loaded successfully")
        
        self.mel_spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )
        self.audio_queue = queue.Queue()
        self.prediction_queue = queue.Queue()
        
        self.command_actions = {
            'Open_Youtube_on_Brave': lambda: self._open_browser_url("brave", "youtube.com"),
            'Open_Gmail_on_Brave': lambda: self._open_browser_url("brave", "gmail.com"),
            'Create_new_Word_Document': lambda: self._open_word(),
            'stop': self.stop_listening,
        }

        self.is_listening = False
        self.energy_threshold = 0.01
        
        print("\nInitialized with commands:", self.commands)
        print("\nMapped actions for:", list(self.command_actions.keys()))
    
    def _load_command_labels(self, data_dir):
        """Load command labels from the data directory"""
        commands = []
        data_path = Path(data_dir)
        if data_path.exists() and data_path.is_dir():
            for folder in data_path.iterdir():
                if folder.is_dir():
                    commands.append(folder.name)
        
        if not commands:
            print("Warning: No command folders found in data directory!")
            commands = ['Open_Youtube_on_Brave', 'Open_Gmail_on_Brave', 'Create_new_Word_Document']
        
        return sorted(commands)
    
    def _open_browser_url(self, browser, url):
        """Open a URL in the specified browser"""
        try:
            if browser.lower() == "brave":
                # Adjust the path to your Brave browser executable
                brave_path = r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"
                if os.path.exists(brave_path):
                    subprocess.Popen([brave_path, f"https://{url}"])
                    print(f"Opening {url} in Brave browser")
                else:
                    print(f"Brave browser not found at {brave_path}")
            else:
                # Default to system default browser
                os.system(f"start https://{url}")
                print(f"Opening {url} in default browser")
        except Exception as e:
            print(f"Error opening browser: {e}")
    
    def _open_word(self):
        """Open a new Word document"""
        try:
            word = win32com.client.Dispatch("Word.Application")
            word.Visible = True
            word.Documents.Add()
            print("Created new Word document")
        except:
            try:
                subprocess.Popen(["start", "winword"])
                print("Started Microsoft Word")
            except Exception as e:
                print(f"Error opening Word: {e}")
    
    def execute_action(self, message):
        print(f"Action: {message}")
    
    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Status: {status}")
        self.audio_queue.put(indata.copy())
        
        if self.debug_mode:
            energy = np.mean(np.abs(indata))
            if energy > self.energy_threshold:
                print(f"Audio detected! Energy level: {energy:.4f}")
    
    def process_audio(self):
        while self.is_listening:
            try:
                audio_data = self.audio_queue.get(timeout=1)
                energy = np.mean(np.abs(audio_data))
                
                if energy < self.energy_threshold:
                    continue
                
                if self.debug_mode:
                    print("\nProcessing audio input...")
                
                # audio to waveform tensor
                waveform = torch.FloatTensor(audio_data.flatten())
                
                target_length = int(self.sample_rate * self.window_duration)
                if waveform.size(0) < target_length:
                    waveform = torch.nn.functional.pad(waveform, (0, target_length - waveform.size(0)))
                else:
                    excess = waveform.size(0) - target_length
                    start = excess // 2
                    waveform = waveform[start:start + target_length]
                    
                waveform = waveform.unsqueeze(0)
                
                # mel spectrogram
                with torch.no_grad():
                    mel_spec = self.mel_spec_transform(waveform)
                    mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
                    mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
                    mel_spec = mel_spec.unsqueeze(0).to(self.device)
                    
                    # Pass through Conformer model
                    output = self.model(mel_spec)
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    
                    if self.debug_mode:
                        print(f"Prediction index: {predicted.item()}")
                        print(f"Confidence: {confidence.item():.4f}")
                        predicted_command = self.commands[predicted.item()]
                        print(f"Predicted command: {predicted_command}")
                    
                    # Higher confidence threshold to reduce false positives
                    if confidence.item() > 0.6:
                        self.prediction_queue.put(predicted.item())
            
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing audio: {e}")
                import traceback
                print(traceback.format_exc())
    
    def execute_commands(self):
        while self.is_listening:
            try:
                predicted_idx = self.prediction_queue.get(timeout=1)
                command = self.commands[predicted_idx]
                print(f"\nRecognized command: {command}")
                
                if command in self.command_actions:
                    self.command_actions[command]()
                else:
                    print(f"No action mapped for command: {command}")
            
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error executing command: {e}")
    
    def stop_listening(self):
        """Stop the voice command system"""
        print("\nStopping voice command system...")
        self.is_listening = False
    
    def start(self):
        """Start the voice command system"""
        print("\nStarting Conformer-based voice command system...")
        print("Available commands:", list(self.command_actions.keys()))
        print("Press 'q' to quit")
        print("\nListening for commands...")
        
        self.is_listening = True
        
        try:
            # Increase blocksize for longer audio capture
            stream = sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                callback=self.audio_callback,
                blocksize=self.window_samples,
                latency='high'  # Use high latency for more stable longer recordings
            )
            processing_thread = threading.Thread(target=self.process_audio)
            command_thread = threading.Thread(target=self.execute_commands)
            
            with stream:
                processing_thread.start()
                command_thread.start()
                
                while self.is_listening:
                    if keyboard.is_pressed('q'):
                        self.stop_listening()
                    time.sleep(0.1)
            
            processing_thread.join()
            command_thread.join()
            print("Voice command system stopped.")
            
        except Exception as e:
            print(f"Error in voice command system: {e}")
            import traceback
            print(traceback.format_exc())
            self.stop_listening()

def main():
    try:
        print("Starting Conformer-based voice command recognition system...")
        print("Make sure your microphone is working and is the default input device")
        
        model_path = "conformer_best_model.pth"
        
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            print("Please run trainConformer.py first to train the Conformer model")
            return
        
        # Start the voice command system
        system = ConformerVoiceCommandSystem(model_path)
        system.start()
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()