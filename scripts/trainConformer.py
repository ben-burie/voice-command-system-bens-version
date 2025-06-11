import os
import torch
import torchaudio
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import random
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
import math

# ------------------------ Dataset Class ------------------------
class BarkVoiceCommandDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None, sample_rate=16000):
        self.root_dir = Path(root_dir)
        self.mode = mode
        self.transform = transform
        self.target_sample_rate = sample_rate
        self.max_duration = 4.0
        
        self.mel_spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )
        
        # 80/20 for train/val
        self.files = []
        self.labels = []
        
        for label_dir in self.root_dir.iterdir():
            if label_dir.is_dir():
                audio_files = list(label_dir.glob('*.wav'))
                if mode == 'train':
                    # 80% for training
                    selected_files = [f for i, f in enumerate(audio_files) if hash(str(f)) % 10 < 8]
                else:  # val mode
                    # 20% for validation
                    selected_files = [f for i, f in enumerate(audio_files) if hash(str(f)) % 10 >= 8]
                
                for audio_file in selected_files:
                    self.files.append(audio_file)
                    self.labels.append(label_dir.name)

        self.unique_labels = sorted(set(self.labels))
        self.label_to_index = {label: i for i, label in enumerate(self.unique_labels)}
        
        print(f"Loaded {len(self.files)} files for {mode} set")
        print(f"Number of classes: {len(self.unique_labels)}")
        print(f"Classes: {self.unique_labels}")
    
    def _load_audio(self, file_path):
        waveform, sample_rate = sf.read(str(file_path))
        waveform = torch.tensor(waveform, dtype=torch.float32)
        
        if len(waveform.shape) > 1:
            waveform = waveform.mean(dim=1)
        
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                sample_rate, 
                self.target_sample_rate
            )
            waveform = resampler(waveform)
        
        target_length = int(self.target_sample_rate * self.max_duration)
        if waveform.size(0) < target_length:
            # pad shorter audio to the target length
            waveform = torch.nn.functional.pad(waveform, (0, target_length - waveform.size(0)))
        else:
            # either trim or use center crop for longer ones
            excess = waveform.size(0) - target_length
            start = excess // 2
            waveform = waveform[start:start + target_length]
        
        return waveform
    
    def _add_noise(self, waveform, noise_level=0.005):
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise
    
    def _time_shift(self, waveform, shift_limit=0.1):
        shift = int(random.random() * shift_limit * len(waveform))
        return torch.roll(waveform, shifts=shift)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        audio_path = self.files[idx]
        label = self.labels[idx]
        
        waveform = self._load_audio(audio_path)
        
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        if self.mode == 'train':
            if random.random() > 0.5:
                waveform = self._add_noise(waveform)
            if random.random() > 0.5:
                waveform = self._time_shift(waveform)
        
        with torch.no_grad():
            mel_spec = self.mel_spec_transform(waveform)
            mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        
        if self.transform:
            mel_spec = self.transform(mel_spec)
        
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        label_idx = self.label_to_index[label]
        
        return mel_spec, label_idx

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
        
        self.d_model = d_model  # d_model as instance variable
        
        # convolutional subsampling to convert mel spectrogram to sequence
        self.conv_subsampling = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU()
        )
        
        # input projection
        self.input_proj = nn.Linear(64, d_model)
        
        # positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # conformer blocks
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(d_model, d_ff, num_heads, kernel_size, dropout)
            for _ in range(num_layers)
        ])
        
        # classification head
        self.classifier = nn.Linear(d_model, num_classes)
        
        # global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        # x shape: (batch, channels, freq, time) = (batch, 1, 64, 64)
        batch_size = x.size(0)
        
        # convolutional subsampling
        x = self.conv_subsampling(x)  # (batch, 64, H, W)
        
        # reshape for sequence processing
        x = x.permute(0, 2, 3, 1)  # (batch, H, W, 64)
        x = x.reshape(batch_size, -1, x.size(3))  # (batch, seq_len, feature_dim)
        
        # project to model dimension
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        
        # positional encoding
        x = self.pos_encoding(x)
        
        # Conformer blocks
        for block in self.conformer_blocks:
            x = block(x)
        
        # global pooling
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x = self.global_pool(x).squeeze(-1)  # (batch, d_model)
        
        # classification
        x = self.classifier(x)
        
        return x

# ------------------------ Training Functions ------------------------
def train_model(model, train_loader, val_loader, device, num_epochs=30):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0

    #------ TRAINING PHASE
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for spectrograms, labels in pbar:
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)
            
            loss.backward()
            # gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100 * train_correct / train_total
        
        #------ VALIDATION PHASE
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for spectrograms, labels in tqdm(val_loader, desc='[Validation]'):
                spectrograms, labels = spectrograms.to(device), labels.to(device)
                
                outputs = model(spectrograms)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100 * val_correct / val_total
        
        scheduler.step(epoch_val_loss)
        
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), 'conformer_best_model.pth')
            print(f"New best model saved with validation accuracy: {best_val_acc:.2f}%")
        
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%')
        print(f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%')
    
    return history

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Validation')
    ax1.set_title('Loss vs. Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['val_acc'], label='Validation')
    ax2.set_title('Accuracy vs. Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('conformer_training_history.png')
    plt.show()

# ------------------------ Main Function ------------------------
def main():
    dataset_path = r"data_barkAI_large"

    print("Creating dataloaders...")
    train_dataset = BarkVoiceCommandDataset(dataset_path, mode='train')
    val_dataset = BarkVoiceCommandDataset(dataset_path, mode='val')
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Conformer model with the number of classes from dataset
    num_classes = len(train_dataset.unique_labels)
    model = ConformerModel(num_classes=num_classes).to(device)
    
    print("\nModel architecture:")
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\nStarting training...")
    history = train_model(model, train_loader, val_loader, device)
    
    plot_training_history(history)

if __name__ == "__main__":
    main()