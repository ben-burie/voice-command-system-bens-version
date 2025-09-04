import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchaudio
import numpy as np
from pathlib import Path
import json
import pickle
from datetime import datetime
from typing import Dict, List, Tuple
import copy
import math

from voiceCommandConformer import ConformerModel
from trainConformer import BarkVoiceCommandDataset

class AdaptiveIncrementalTrainer:
    def __init__(self, model_path: str, base_data_dir: str, config_path: str):
        self.model_path = Path(model_path)
        self.base_data_dir = Path(base_data_dir)
        self.config_path = Path(config_path)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.config = self._load_config()
        
        self.model = None
        self.optimizer_state = None
        self.class_mapping = {}
        
        print(f"Adaptive Incremental Trainer initialized")
        print(f"Device: {self.device}")
    
    def _load_config(self) -> Dict:
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            return {
                "adaptive_settings": {
                    "base_learning_rate": 0.0001,
                    "fine_tune_epochs": 5,
                    "freeze_layers": True,
                    
                    "adaptive_freezing": True,
                    "min_unfreeze_layers": 1,
                    "max_unfreeze_layers": 4,
                    
                    "freezing_strategy": {
                        "small_model": {"commands": 5, "unfreeze_ratio": 0.3},
                        "medium_model": {"commands": 10, "unfreeze_ratio": 0.4},
                        "large_model": {"commands": 20, "unfreeze_ratio": 0.5},
                        "very_large_model": {"commands": 50, "unfreeze_ratio": 0.6}
                    },
            
                    "progressive_unfreezing": {
                        "enabled": True,
                        "start_epoch": 2,
                        "unfreeze_per_epoch": 0.5
                    }
                },
                "class_history": [],
                "training_history": []
            }
    
    def _save_config(self):
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def _calculate_optimal_freezing_strategy(self, num_commands: int, total_blocks: int) -> Dict:
        strategy_config = self.config["adaptive_settings"]["freezing_strategy"]
        
        if num_commands <= strategy_config["small_model"]["commands"]:
            category = "small_model"
        elif num_commands <= strategy_config["medium_model"]["commands"]:
            category = "medium_model"
        elif num_commands <= strategy_config["large_model"]["commands"]:
            category = "large_model"
        else:
            category = "very_large_model"
        
        unfreeze_ratio = strategy_config[category]["unfreeze_ratio"]
        unfreeze_layers = max(
            self.config["adaptive_settings"]["min_unfreeze_layers"],
            min(
                self.config["adaptive_settings"]["max_unfreeze_layers"],
                int(total_blocks * unfreeze_ratio)
            )
        )
        
        base_lr = self.config["adaptive_settings"]["base_learning_rate"]
        lr_scale = 1.0 / (1.0 + 0.1 * math.log(num_commands))
        adjusted_lr = base_lr * lr_scale
        base_epochs = self.config["adaptive_settings"]["fine_tune_epochs"]
        epoch_scale = 1.0 + 0.2 * math.log(num_commands / 5.0) if num_commands > 5 else 1.0
        adjusted_epochs = int(base_epochs * epoch_scale)
        
        strategy = {
            "category": category,
            "unfreeze_layers": unfreeze_layers,
            "unfreeze_ratio": unfreeze_ratio,
            "learning_rate": adjusted_lr,
            "epochs": adjusted_epochs,
            "num_commands": num_commands,
            "total_blocks": total_blocks
        }
        
        print(f"\n=== Adaptive Freezing Strategy ===")
        print(f"Model category: {category}")
        print(f"Commands: {num_commands}")
        print(f"Total conformer blocks: {total_blocks}")
        print(f"Unfreezing last {unfreeze_layers} blocks ({unfreeze_ratio:.1%} of model)")
        print(f"Adjusted learning rate: {adjusted_lr:.6f}")
        print(f"Adjusted epochs: {adjusted_epochs}")
        
        return strategy
    
    def load_existing_model(self) -> bool:
        try:
            if not self.model_path.exists():
                print("No existing model found")
                return False
            
            current_classes = self._get_current_classes()
            num_classes = len(current_classes)
            self.model = ConformerModel(num_classes=num_classes).to(self.device)
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model_state = checkpoint['model_state_dict']
                optimizer_state = checkpoint.get('optimizer_state_dict')
                class_mapping = checkpoint.get('class_mapping', {})
            else:
                model_state = checkpoint
                optimizer_state = None
                class_mapping = {}
    
            old_num_classes = model_state['classifier.weight'].shape[0]
            
            if old_num_classes != num_classes:
                print(f"Expanding model from {old_num_classes} to {num_classes} classes")
                self.model = self._expand_model_classes(self.model, model_state, old_num_classes, num_classes)
                # Clear optimizer state when model structure changes to avoid tensor size mismatch
                self.optimizer_state = None
                print("Cleared optimizer state due to model expansion")
            else:
                self.model.load_state_dict(model_state)
                self.optimizer_state = optimizer_state
            
            self.class_mapping = class_mapping
            
            print(f"Loaded model with {num_classes} classes")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def _get_current_classes(self) -> List[str]:
        classes = []
        for folder in self.base_data_dir.iterdir():
            if folder.is_dir():
                classes.append(folder.name)
        return sorted(classes)
    
    def _expand_model_classes(self, model: ConformerModel, old_state: Dict, 
                            old_classes: int, new_classes: int) -> ConformerModel:
        model_dict = model.state_dict()
        for key, value in old_state.items():
            if key.startswith('classifier.'):
                continue
            if key in model_dict:
                model_dict[key] = value
    
        old_weight = old_state['classifier.weight']  # [old_classes, d_model]
        old_bias = old_state['classifier.bias']      # [old_classes]
        
        new_weight = model_dict['classifier.weight']  # [new_classes, d_model]
        new_bias = model_dict['classifier.bias']      # [new_classes]
        
        new_weight[:old_classes] = old_weight
        new_bias[:old_classes] = old_bias
        
        nn.init.xavier_uniform_(new_weight[old_classes:])
        nn.init.zeros_(new_bias[old_classes:])
        
        model_dict['classifier.weight'] = new_weight
        model_dict['classifier.bias'] = new_bias
        
        model.load_state_dict(model_dict)
        return model
    
    def prepare_incremental_training(self, new_command_dirs: List[str]) -> Tuple[bool, Dict]:
        try:
            if not self.load_existing_model():
                print("Creating new model...")
                current_classes = self._get_current_classes()
                self.model = ConformerModel(num_classes=len(current_classes)).to(self.device)
            
            num_commands = len(self._get_current_classes())
            total_blocks = len(self.model.conformer_blocks)
            
            strategy = self._calculate_optimal_freezing_strategy(num_commands, total_blocks)
            
            if self.config["adaptive_settings"]["freeze_layers"]:
                self._adaptive_freeze_model_layers(strategy)
            
            return True, strategy
            
        except Exception as e:
            print(f"Error preparing incremental training: {e}")
            return False, {}
    
    def _adaptive_freeze_model_layers(self, strategy: Dict):
        unfreeze_layers = strategy["unfreeze_layers"]
        total_blocks = strategy["total_blocks"]
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        
        unfreeze_from = max(0, total_blocks - unfreeze_layers)
        
        for i in range(unfreeze_from, total_blocks):
            for param in self.model.conformer_blocks[i].parameters():
                param.requires_grad = True
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"Adaptive frozen model: {trainable_params:,}/{total_params:,} trainable parameters")
        print(f"Unfrozen blocks: {unfreeze_from} to {total_blocks-1} (last {unfreeze_layers} blocks)")
    
    def _progressive_unfreeze(self, epoch: int, strategy: Dict):
        if not self.config["adaptive_settings"]["progressive_unfreezing"]["enabled"]:
            return
        
        start_epoch = self.config["adaptive_settings"]["progressive_unfreezing"]["start_epoch"]
        unfreeze_per_epoch = self.config["adaptive_settings"]["progressive_unfreezing"]["unfreeze_per_epoch"]
        
        if epoch >= start_epoch:
            total_blocks = len(self.model.conformer_blocks)
            current_unfrozen = strategy["unfreeze_layers"]
            
            additional_layers = int((epoch - start_epoch + 1) * unfreeze_per_epoch)
            new_unfrozen = min(total_blocks, current_unfrozen + additional_layers)
            
            if new_unfrozen > current_unfrozen:
                print(f"Progressive unfreezing: unfreezing {new_unfrozen} layers (was {current_unfrozen})")
                unfreeze_from = max(0, total_blocks - new_unfrozen)
                for i in range(unfreeze_from, total_blocks - current_unfrozen):
                    for param in self.model.conformer_blocks[i].parameters():
                        param.requires_grad = True
                
                strategy["unfreeze_layers"] = new_unfrozen
    
    def incremental_train(self, new_command_dirs: List[str], epochs: int = None) -> bool:
        try:
            print(f"\n=== Adaptive Incremental Training ===")
            print(f"New commands: {new_command_dirs}")
            
            success, strategy = self.prepare_incremental_training(new_command_dirs)
            if not success:
                return False
            
            if epochs is None:
                epochs = strategy["epochs"]
            
            train_dataset = BarkVoiceCommandDataset(str(self.base_data_dir), mode='train')
            val_dataset = BarkVoiceCommandDataset(str(self.base_data_dir), mode='val')
            batch_size = max(4, min(16, 32 // max(1, strategy["num_commands"] // 10)))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
            
            lr = strategy["learning_rate"]
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()), 
                lr=lr, 
                weight_decay=1e-5
            )
            
            # Check if model was expanded - if so, don't load old optimizer state
            current_classes = len(self._get_current_classes())
            if self.optimizer_state:
                try:
                    # Check if optimizer state matches current model structure
                    old_classifier_weight_shape = None
                    for group in self.optimizer_state['state'].values():
                        if 'exp_avg' in group:
                            # Find classifier layer state by checking tensor shapes
                            exp_avg_shape = group['exp_avg'].shape
                            if len(exp_avg_shape) == 1:  # This could be classifier bias
                                old_classifier_weight_shape = exp_avg_shape[0]
                                break
                    
                    # Only load optimizer state if model structure hasn't changed
                    if old_classifier_weight_shape is None or old_classifier_weight_shape == current_classes:
                        optimizer.load_state_dict(self.optimizer_state)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                        print("Loaded existing optimizer state")
                    else:
                        print(f"Model expanded from {old_classifier_weight_shape} to {current_classes} classes - using fresh optimizer")
                except Exception as e:
                    print(f"Could not load optimizer state ({e}), using fresh optimizer")
            
            criterion = nn.CrossEntropyLoss()
            
            best_val_acc = 0
            for epoch in range(epochs):
                self._progressive_unfreeze(epoch, strategy)
                
                # Training phase
                self.model.train()
                train_loss = 0
                train_correct = 0
                train_total = 0
                
                for batch_idx, (spectrograms, labels) in enumerate(train_loader):
                    spectrograms, labels = spectrograms.to(self.device), labels.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(spectrograms)
                    loss = criterion(outputs, labels)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()
                    
                    if batch_idx % 10 == 0:
                        print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                
                # Validation phase
                self.model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for spectrograms, labels in val_loader:
                        spectrograms, labels = spectrograms.to(self.device), labels.to(self.device)
                        outputs = self.model(spectrograms)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                
                # Calculate metrics
                epoch_train_acc = 100 * train_correct / train_total
                epoch_val_acc = 100 * val_correct / val_total
                
                print(f"Epoch {epoch+1}/{epochs}:")
                print(f"  Train Acc: {epoch_train_acc:.2f}%, Val Acc: {epoch_val_acc:.2f}%")
                print(f"  Unfrozen layers: {strategy['unfreeze_layers']}")
                
                # Save best model
                if epoch_val_acc > best_val_acc:
                    best_val_acc = epoch_val_acc
                    self._save_model(optimizer, train_dataset.unique_labels, strategy)
                    print(f"   New best model saved (Val Acc: {best_val_acc:.2f}%)")

            self.config["training_history"].append({
                "timestamp": datetime.now().isoformat(),
                "type": "adaptive_incremental",
                "new_commands": new_command_dirs,
                "epochs": epochs,
                "best_val_acc": best_val_acc,
                "strategy_used": strategy
            })
            self._save_config()
            
            print(f"\nAdaptive incremental training completed!")
            print(f"Best validation accuracy: {best_val_acc:.2f}%")
            print(f"Strategy used: {strategy['category']} with {strategy['unfreeze_layers']} unfrozen layers")
            return True
            
        except Exception as e:
            print(f"Error in adaptive incremental training: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _save_model(self, optimizer, class_labels, strategy):
        class_mapping = {label: idx for idx, label in enumerate(class_labels)}
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'class_mapping': class_mapping,
            'timestamp': datetime.now().isoformat(),
            'num_classes': len(class_labels),
            'training_strategy': strategy
        }
        
        torch.save(checkpoint, self.model_path)
        torch.save(self.model.state_dict(), str(self.model_path).replace('.pth', '_state_dict.pth'))
    
    def quick_add_command(self, command_name: str, audio_files: List[str]) -> bool:
        try:
            print(f"Quick-adding command with adaptive strategy: {command_name}")
            command_dir = self.base_data_dir / command_name
            command_dir.mkdir(exist_ok=True)
            success = self.incremental_train([command_name])
            
            if success:
                print(f"Command '{command_name}' added and trained successfully with adaptive strategy!")
            
            return success
            
        except Exception as e:
            print(f"Error quick-adding command: {e}")
            return False
    
    def analyze_training_history(self):
        print("\n=== Training History Analysis ===")
        
        for i, entry in enumerate(self.config["training_history"]):
            if entry["type"] == "adaptive_incremental":
                strategy = entry.get("strategy_used", {})
                print(f"\nTraining {i+1}: {entry['timestamp'][:10]}")
                print(f"  Commands added: {entry['new_commands']}")
                print(f"  Total commands: {strategy.get('num_commands', 'N/A')}")
                print(f"  Strategy: {strategy.get('category', 'N/A')}")
                print(f"  Unfrozen layers: {strategy.get('unfreeze_layers', 'N/A')}")
                print(f"  Learning rate: {strategy.get('learning_rate', 'N/A'):.6f}")
                print(f"  Final accuracy: {entry['best_val_acc']:.2f}%")


class SmartRetrainingScheduler:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        return {"retraining_rules": {}}
    
    def should_full_retrain(self, new_commands_count: int, total_commands: int, 
                          last_retrain_date: str = None) -> bool:
        """Decide whether to do full retraining or incremental training"""
        
        # Rule 1: If adding more than 20% new commands, do full retrain
        if new_commands_count / max(total_commands, 1) > 0.2:
            return True
        
        # Rule 2: If more than 10 new commands at once, do full retrain
        if new_commands_count > 10:
            return True
        
        # Rule 3: If haven't done full retrain in 30 days and have 5+ new commands
        if last_retrain_date:
            from datetime import datetime, timedelta
            last_retrain = datetime.fromisoformat(last_retrain_date)
            if (datetime.now() - last_retrain).days > 30 and new_commands_count >= 5:
                return True
        
        # Default to incremental training
        return False


def main():
    """Demo of adaptive incremental training system"""
    print("=== Adaptive Incremental Training System Demo ===")
    
    trainer = AdaptiveIncrementalTrainer(
        model_path="models/conformer_best_model.pth",
        base_data_dir="data_barkAI_large",
        config_path="config/adaptive_incremental_config.json"
    )
    
    # Analyze current training history
    trainer.analyze_training_history()
    
    # Example: Add a new command with adaptive strategy
    # trainer.quick_add_command("Open_Discord", [])
    
    print("Adaptive incremental trainer ready!")


if __name__ == "__main__":
    main()
