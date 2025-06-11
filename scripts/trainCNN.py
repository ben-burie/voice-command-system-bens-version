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
import time
import json

# ------------------------ Dataset Class (Same as Conformer) ------------------------
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

# ------------------------ CNN Model ------------------------
class CNNModel(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super(CNNModel, self).__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # Adaptive pooling to fixed size
            nn.Dropout2d(dropout)
        )
        
        # Calculate the size after conv layers
        # After adaptive pooling: 256 * 4 * 4 = 4096
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, 1, freq, time)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

# ------------------------ Training Functions ------------------------
def train_model(model, train_loader, val_loader, device, num_epochs=30):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0
    start_time = time.time()

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
            torch.save(model.state_dict(), 'cnn_best_model.pth')
            print(f"New best CNN model saved with validation accuracy: {best_val_acc:.2f}%")
        
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%')
        print(f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%')
    
    total_time = time.time() - start_time
    history['total_training_time'] = total_time
    history['best_val_acc'] = best_val_acc
    
    return history

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Validation')
    ax1.set_title('CNN Loss vs. Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['val_acc'], label='Validation')
    ax2.set_title('CNN Accuracy vs. Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('cnn_training_history.png')
    plt.show()

def save_model_info(model, history, dataset):
    """Save model information for comparison"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    model_info = {
        'model_type': 'CNN',
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'best_val_accuracy': history['best_val_acc'],
        'total_training_time': history['total_training_time'],
        'num_classes': len(dataset.unique_labels),
        'classes': dataset.unique_labels,
        'final_train_acc': history['train_acc'][-1],
        'final_val_acc': history['val_acc'][-1],
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1]
    }
    
    with open('cnn_model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"\nCNN Model Information:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Best validation accuracy: {history['best_val_acc']:.2f}%")
    print(f"Total training time: {history['total_training_time']:.2f} seconds")

# ------------------------ Main Function ------------------------
def main():
    dataset_path = r"data_barkAI_large"

    print("Creating CNN dataloaders...")
    train_dataset = BarkVoiceCommandDataset(dataset_path, mode='train')
    val_dataset = BarkVoiceCommandDataset(dataset_path, mode='val')
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # CNN model with the number of classes from dataset
    num_classes = len(train_dataset.unique_labels)
    model = CNNModel(num_classes=num_classes).to(device)
    
    print("\nCNN Model architecture:")
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\nStarting CNN training...")
    history = train_model(model, train_loader, val_loader, device)
    
    plot_training_history(history)
    save_model_info(model, history, train_dataset)

if __name__ == "__main__":
    main()
