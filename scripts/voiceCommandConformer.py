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
from scipy import signal
from scipy.ndimage import uniform_filter1d
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
    def __init__(self, model_path, sample_rate=16000, window_duration=4, action_registry=None):
        self.debug_mode = True
        self.sample_rate = sample_rate
        self.window_duration = window_duration
        self.window_samples = int(sample_rate * window_duration)
        
        print("Initializing Conformer model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load the command labels from the data directory
        self.commands = self._load_command_labels("data_barkAI_large")
        num_classes = len(self.commands)
        print(f"Detected {num_classes} command classes: {self.commands}")
        
        self.model = ConformerModel(num_classes=num_classes).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # New format with metadata
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded model from checkpoint with metadata")
        else:
            # Old format - direct state dict
            self.model.load_state_dict(checkpoint)
            print("Loaded model from direct state dict")
        
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
        
        # Default command actions
        self.command_actions = {
            'Open_Youtube_on_Brave': lambda: self._open_browser_url("brave", "youtube.com"),
            'Open_Gmail_on_Brave': lambda: self._open_browser_url("brave", "gmail.com"),
            'Create_new_Word_Document': lambda: self._open_word(),
            'stop': self.stop_listening,
        }
        
        # Merge with external action registry if provided
        if action_registry:
            self.command_actions.update(action_registry)

        self.is_listening = False
        self.is_paused = False  # Add pause functionality
        
        # Enhanced audio processing parameters
        self.energy_threshold = 0.005  # Lowered threshold for low audio
        self.adaptive_threshold = True
        self.auto_gain_control = True
        self.noise_reduction = True
        self.audio_enhancement = True
        
        # Audio enhancement parameters
        self.gain_factor = 2.0  # Initial gain boost
        self.max_gain = 10.0    # Maximum gain limit
        self.noise_floor = 0.001  # Noise floor estimation
        self.smoothing_factor = 0.95  # For adaptive threshold
        self.running_energy_avg = 0.01  # Running average of energy
        
        # Audio buffer for better processing
        self.audio_buffer = []
        self.buffer_size = 3  # Number of audio chunks to buffer
        
        print("\nAudio Enhancement Features Enabled:")
        print(f"  - Adaptive Gain Control: {self.auto_gain_control}")
        print(f"  - Noise Reduction: {self.noise_reduction}")
        print(f"  - Audio Enhancement: {self.audio_enhancement}")
        print(f"  - Initial Gain Factor: {self.gain_factor}x")
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
            if not url.startswith(('http://', 'https://')):
                url = f"https://{url}"
            
            if browser.lower() == "brave":
                brave_path = r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"
                if os.path.exists(brave_path):
                    subprocess.Popen([brave_path, url])
                    print(f"Opening {url} in Brave browser")
                else:
                    print(f"Brave browser not found, using default browser")
                    os.system(f"start {url}")
            else:
                os.system(f"start {url}")
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
    
    def _open_application(self, app_path):
        """Open an application"""
        try:
            subprocess.Popen(app_path, shell=True)
            print(f"Opening application: {app_path}")
        except Exception as e:
            print(f"Error opening application: {e}")
    
    def _execute_system_command(self, cmd):
        """Execute system command"""
        try:
            subprocess.run(cmd, shell=True)
            print(f"Executed: {cmd}")
        except Exception as e:
            print(f"Error executing command: {e}")
    
    def _enhance_audio(self, audio_data):
        """
        Comprehensive audio enhancement for low input signals
        """
        try:
            # Convert to numpy array if needed
            if isinstance(audio_data, torch.Tensor):
                audio_data = audio_data.numpy()
            
            audio_data = audio_data.flatten()
            original_energy = np.mean(np.abs(audio_data))
            
            if self.debug_mode and original_energy > self.noise_floor:
                print(f"Original audio energy: {original_energy:.6f}")
            
            # 1. Noise reduction using spectral subtraction
            if self.noise_reduction:
                audio_data = self._spectral_noise_reduction(audio_data)
            
            # 2. Automatic Gain Control (AGC)
            if self.auto_gain_control:
                audio_data = self._apply_automatic_gain_control(audio_data)
            
            # 3. Dynamic range compression
            if self.audio_enhancement:
                audio_data = self._apply_dynamic_compression(audio_data)
            
            # 4. High-pass filter to remove low-frequency noise
            audio_data = self._apply_highpass_filter(audio_data)
            
            # 5. Adaptive threshold update
            if self.adaptive_threshold:
                self._update_adaptive_threshold(original_energy)
            
            enhanced_energy = np.mean(np.abs(audio_data))
            
            if self.debug_mode and enhanced_energy > self.noise_floor:
                enhancement_ratio = enhanced_energy / (original_energy + 1e-8)
                print(f"Enhanced audio energy: {enhanced_energy:.6f} (boost: {enhancement_ratio:.2f}x)")
            
            return audio_data
            
        except Exception as e:
            print(f"Error in audio enhancement: {e}")
            return audio_data.flatten() if hasattr(audio_data, 'flatten') else audio_data
    
    def _spectral_noise_reduction(self, audio_data):
        """
        Apply spectral subtraction for noise reduction
        """
        try:
            # Compute STFT
            f, t, stft = signal.stft(audio_data, fs=self.sample_rate, nperseg=512)
            
            # Estimate noise spectrum from first few frames (assumed to be noise)
            noise_frames = min(5, stft.shape[1] // 4)
            noise_spectrum = np.mean(np.abs(stft[:, :noise_frames]), axis=1, keepdims=True)
            
            # Apply spectral subtraction
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Subtract noise spectrum with over-subtraction factor
            alpha = 2.0  # Over-subtraction factor
            beta = 0.01  # Spectral floor factor
            
            enhanced_magnitude = magnitude - alpha * noise_spectrum
            enhanced_magnitude = np.maximum(enhanced_magnitude, beta * magnitude)
            
            # Reconstruct signal
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            _, enhanced_audio = signal.istft(enhanced_stft, fs=self.sample_rate, nperseg=512)
            
            return enhanced_audio[:len(audio_data)]  # Ensure same length
            
        except Exception as e:
            if self.debug_mode:
                print(f"Spectral noise reduction failed: {e}")
            return audio_data
    
    def _apply_automatic_gain_control(self, audio_data):
        """
        Apply automatic gain control to boost low signals
        """
        try:
            # Calculate RMS energy
            rms_energy = np.sqrt(np.mean(audio_data ** 2))
            
            if rms_energy < self.noise_floor:
                return audio_data
            
            # Target RMS level
            target_rms = 0.1
            
            # Calculate required gain
            required_gain = target_rms / (rms_energy + 1e-8)
            
            # Limit gain to prevent over-amplification
            gain = min(required_gain, self.max_gain)
            gain = max(gain, 1.0)  # Don't reduce signal
            
            # Apply gain with soft limiting
            enhanced_audio = audio_data * gain
            
            # Soft limiting to prevent clipping
            enhanced_audio = np.tanh(enhanced_audio * 0.9) / 0.9
            
            if self.debug_mode and gain > 1.1:
                print(f"AGC applied gain: {gain:.2f}x")
            
            return enhanced_audio
            
        except Exception as e:
            if self.debug_mode:
                print(f"AGC failed: {e}")
            return audio_data
    
    def _apply_dynamic_compression(self, audio_data):
        """
        Apply dynamic range compression to enhance quiet signals
        """
        try:
            # Parameters for compression
            threshold = 0.1
            ratio = 4.0
            attack_time = 0.003  # 3ms
            release_time = 0.1   # 100ms
            
            # Convert time constants to samples
            attack_samples = int(attack_time * self.sample_rate)
            release_samples = int(release_time * self.sample_rate)
            
            # Calculate envelope
            envelope = np.abs(audio_data)
            
            # Smooth envelope
            for i in range(1, len(envelope)):
                if envelope[i] > envelope[i-1]:
                    # Attack
                    alpha = 1.0 - np.exp(-1.0 / attack_samples)
                else:
                    # Release
                    alpha = 1.0 - np.exp(-1.0 / release_samples)
                
                envelope[i] = alpha * envelope[i] + (1 - alpha) * envelope[i-1]
            
            # Apply compression
            gain = np.ones_like(envelope)
            over_threshold = envelope > threshold
            
            if np.any(over_threshold):
                gain[over_threshold] = threshold / envelope[over_threshold]
                gain[over_threshold] = gain[over_threshold] ** (1.0 / ratio - 1.0)
            
            # Apply makeup gain for signals below threshold
            below_threshold = envelope <= threshold
            if np.any(below_threshold):
                makeup_gain = 2.0  # Boost quiet signals
                gain[below_threshold] *= makeup_gain
            
            # Smooth gain changes
            gain = uniform_filter1d(gain, size=int(0.01 * self.sample_rate))
            
            return audio_data * gain
            
        except Exception as e:
            if self.debug_mode:
                print(f"Dynamic compression failed: {e}")
            return audio_data
    
    def _apply_highpass_filter(self, audio_data):
        """
        Apply high-pass filter to remove low-frequency noise
        """
        try:
            # Design high-pass filter (remove frequencies below 80 Hz)
            nyquist = self.sample_rate / 2
            cutoff = 80 / nyquist
            
            # Use a 4th order Butterworth filter
            b, a = signal.butter(4, cutoff, btype='high')
            
            # Apply filter
            filtered_audio = signal.filtfilt(b, a, audio_data)
            
            return filtered_audio
            
        except Exception as e:
            if self.debug_mode:
                print(f"High-pass filter failed: {e}")
            return audio_data
    
    def _update_adaptive_threshold(self, current_energy):
        """
        Update adaptive energy threshold based on ambient noise
        """
        try:
            # Update running average of energy
            self.running_energy_avg = (self.smoothing_factor * self.running_energy_avg + 
                                     (1 - self.smoothing_factor) * current_energy)
            
            # Set threshold as multiple of running average
            adaptive_threshold = max(self.running_energy_avg * 2.0, 0.001)
            
            # Smooth threshold changes
            self.energy_threshold = (0.9 * self.energy_threshold + 
                                   0.1 * adaptive_threshold)
            
            if self.debug_mode and abs(adaptive_threshold - self.energy_threshold) > 0.001:
                print(f"Adaptive threshold updated: {self.energy_threshold:.6f}")
                
        except Exception as e:
            if self.debug_mode:
                print(f"Adaptive threshold update failed: {e}")
    
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
                
                # Skip audio processing if paused
                if self.is_paused:
                    continue
                
                # Apply audio enhancement for low input signals
                enhanced_audio = self._enhance_audio(audio_data)
                
                energy = np.mean(np.abs(enhanced_audio))
                
                if energy < self.energy_threshold:
                    continue
                
                if self.debug_mode:
                    print("\nProcessing enhanced audio input...")
                
                # Convert enhanced audio to waveform tensor
                waveform = torch.FloatTensor(enhanced_audio.flatten())
                
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
                        predicted_command = self.commands[predicted.item()]
                        print(f"Prediction index: {predicted.item()}")
                        print(f"Confidence: {confidence.item():.4f}")
                        print(f"Predicted command: {predicted_command}")
                    
                    # Simple confidence check
                    if confidence > 0.5:
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
                
                # Skip command execution if paused
                if self.is_paused:
                    continue
                
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
    
    def adjust_audio_settings(self, setting, value):
        """
        Dynamically adjust audio enhancement settings
        """
        try:
            if setting == "gain_factor":
                self.gain_factor = max(1.0, min(value, self.max_gain))
                print(f"Gain factor set to: {self.gain_factor}x")
            elif setting == "energy_threshold":
                self.energy_threshold = max(0.001, min(value, 0.1))
                print(f"Energy threshold set to: {self.energy_threshold}")
            elif setting == "noise_reduction":
                self.noise_reduction = bool(value)
                print(f"Noise reduction: {'enabled' if self.noise_reduction else 'disabled'}")
            elif setting == "auto_gain_control":
                self.auto_gain_control = bool(value)
                print(f"Auto gain control: {'enabled' if self.auto_gain_control else 'disabled'}")
            elif setting == "audio_enhancement":
                self.audio_enhancement = bool(value)
                print(f"Audio enhancement: {'enabled' if self.audio_enhancement else 'disabled'}")
            elif setting == "adaptive_threshold":
                self.adaptive_threshold = bool(value)
                print(f"Adaptive threshold: {'enabled' if self.adaptive_threshold else 'disabled'}")
            else:
                print(f"Unknown setting: {setting}")
                return False
            return True
        except Exception as e:
            print(f"Error adjusting audio setting: {e}")
            return False
    
    def show_audio_settings(self):
        """
        Display current audio enhancement settings
        """
        print("\n" + "="*50)
        print("CURRENT AUDIO ENHANCEMENT SETTINGS")
        print("="*50)
        print(f"Energy Threshold: {self.energy_threshold:.6f}")
        print(f"Gain Factor: {self.gain_factor}x")
        print(f"Max Gain Limit: {self.max_gain}x")
        print(f"Noise Floor: {self.noise_floor:.6f}")
        print(f"Running Energy Average: {self.running_energy_avg:.6f}")
        print("\nFeature Status:")
        print(f"  - Adaptive Threshold: {'✓' if self.adaptive_threshold else '✗'}")
        print(f"  - Auto Gain Control: {'✓' if self.auto_gain_control else '✗'}")
        print(f"  - Noise Reduction: {'✓' if self.noise_reduction else '✗'}")
        print(f"  - Audio Enhancement: {'✓' if self.audio_enhancement else '✗'}")
        print("="*50)
    
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
        
        model_path = "models/conformer_best_model.pth"
        
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
