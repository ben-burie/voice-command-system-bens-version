import os
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import time
from tqdm import tqdm
import math

# Import model architectures
from trainConformer import ConformerModel, BarkVoiceCommandDataset
from trainCNN import CNNModel
from trainLSTM import LSTMModel, BarkVoiceCommandDatasetLSTM

# ------------------------ Model Comparison Class ------------------------
class ModelComparison:
    def __init__(self, dataset_path="../data_barkAI_large"):
        self.dataset_path = dataset_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load datasets
        self.conformer_dataset = BarkVoiceCommandDataset(dataset_path, mode='val')
        self.lstm_dataset = BarkVoiceCommandDatasetLSTM(dataset_path, mode='val')
        
        self.num_classes = len(self.conformer_dataset.unique_labels)
        self.class_names = self.conformer_dataset.unique_labels
        
        # Initialize models
        self.models = {}
        self.model_info = {}
        
        print(f"Initialized comparison for {self.num_classes} classes: {self.class_names}")
    
    def load_models(self):
        """Load all trained models"""
        print("Loading trained models...")
        
        # Load Conformer model
        try:
            self.models['Conformer'] = ConformerModel(num_classes=self.num_classes).to(self.device)
            self.models['Conformer'].load_state_dict(torch.load('../models/conformer_best_model.pth', map_location=self.device))
            self.models['Conformer'].eval()
            print("✓ Conformer model loaded")
        except Exception as e:
            print(f"✗ Failed to load Conformer model: {e}")
        
        # Load CNN model
        try:
            self.models['CNN'] = CNNModel(num_classes=self.num_classes).to(self.device)
            self.models['CNN'].load_state_dict(torch.load('../models/cnn_best_model.pth', map_location=self.device))
            self.models['CNN'].eval()
            print("✓ CNN model loaded")
        except Exception as e:
            print(f"✗ Failed to load CNN model: {e}")
        
        # Load LSTM model
        try:
            self.models['LSTM'] = LSTMModel(num_classes=self.num_classes).to(self.device)
            self.models['LSTM'].load_state_dict(torch.load('../models/lstm_best_model.pth', map_location=self.device))
            self.models['LSTM'].eval()
            print("✓ LSTM model loaded")
        except Exception as e:
            print(f"✗ Failed to load LSTM model: {e}")
    
    def load_training_info(self):
        """Load training information from JSON files"""
        print("Loading training information...")
        
        info_files = {
            'CNN': '../config/cnn_model_info.json',
            'LSTM': '../config/lstm_model_info.json'
        }
        
        for model_name, file_path in info_files.items():
            try:
                with open(file_path, 'r') as f:
                    self.model_info[model_name] = json.load(f)
                print(f"✓ {model_name} training info loaded")
            except Exception as e:
                print(f"✗ Failed to load {model_name} training info: {e}")
        
        # Add Conformer info manually (since we don't have a JSON file for it)
        if 'Conformer' in self.models:
            conformer_params = sum(p.numel() for p in self.models['Conformer'].parameters())
            self.model_info['Conformer'] = {
                'model_type': 'Conformer',
                'total_parameters': conformer_params,
                'trainable_parameters': conformer_params,
                'num_classes': self.num_classes,
                'classes': self.class_names
            }
    
    def evaluate_model_performance(self):
        """Evaluate all models on validation set"""
        print("Evaluating model performance...")
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\nEvaluating {model_name}...")
            
            if model_name == 'LSTM':
                dataset = self.lstm_dataset
            else:
                dataset = self.conformer_dataset
            
            dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
            
            # Performance metrics
            correct = 0
            total = 0
            class_correct = [0] * self.num_classes
            class_total = [0] * self.num_classes
            all_predictions = []
            all_labels = []
            inference_times = []
            
            model.eval()
            with torch.no_grad():
                for data, labels in tqdm(dataloader, desc=f"Evaluating {model_name}"):
                    data, labels = data.to(self.device), labels.to(self.device)
                    
                    # Measure inference time
                    start_time = time.time()
                    outputs = model(data)
                    inference_time = time.time() - start_time
                    inference_times.append(inference_time / data.size(0))  # Per sample
                    
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # Per-class accuracy
                    for i in range(labels.size(0)):
                        label = labels[i]
                        class_correct[label] += (predicted[i] == label).item()
                        class_total[label] += 1
                    
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # Calculate metrics
            accuracy = 100 * correct / total
            avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
            
            per_class_accuracy = {}
            for i in range(self.num_classes):
                if class_total[i] > 0:
                    per_class_accuracy[self.class_names[i]] = 100 * class_correct[i] / class_total[i]
                else:
                    per_class_accuracy[self.class_names[i]] = 0
            
            results[model_name] = {
                'accuracy': accuracy,
                'avg_inference_time_ms': avg_inference_time,
                'per_class_accuracy': per_class_accuracy,
                'predictions': all_predictions,
                'labels': all_labels,
                'total_samples': total
            }
            
            print(f"{model_name} - Accuracy: {accuracy:.2f}%, Avg Inference: {avg_inference_time:.2f}ms")
        
        return results
    
    def measure_model_complexity(self):
        """Measure model complexity metrics"""
        print("Measuring model complexity...")
        
        complexity_results = {}
        
        for model_name, model in self.models.items():
            # Parameter count
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Model size (approximate)
            model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
            
            # Memory usage during inference (approximate)
            model.eval()
            if model_name == 'LSTM':
                dummy_input = torch.randn(1, 125, 13).to(self.device)  # MFCC features
            else:
                dummy_input = torch.randn(1, 1, 64, 125).to(self.device)  # Mel spectrogram
            
            # Measure memory before and after
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                with torch.no_grad():
                    _ = model(dummy_input)
                
                memory_usage_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            else:
                memory_usage_mb = 0  # Can't measure CPU memory easily
            
            complexity_results[model_name] = {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': model_size_mb,
                'memory_usage_mb': memory_usage_mb
            }
        
        return complexity_results
    
    def create_comparison_visualizations(self, performance_results, complexity_results):
        """Create comprehensive comparison visualizations"""
        print("Creating comparison visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create a comprehensive comparison figure
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Overall Accuracy Comparison
        ax1 = plt.subplot(2, 4, 1)
        models = list(performance_results.keys())
        accuracies = [performance_results[model]['accuracy'] for model in models]
        bars = ax1.bar(models, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. Parameter Count Comparison
        ax2 = plt.subplot(2, 4, 2)
        param_counts = [complexity_results[model]['total_parameters']/1e6 for model in models]
        bars = ax2.bar(models, param_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax2.set_title('Model Parameters (Millions)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Parameters (M)')
        
        for bar, params in zip(bars, param_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{params:.2f}M', ha='center', va='bottom', fontweight='bold')
        
        # 3. Inference Time Comparison
        ax3 = plt.subplot(2, 4, 3)
        inference_times = [performance_results[model]['avg_inference_time_ms'] for model in models]
        bars = ax3.bar(models, inference_times, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax3.set_title('Average Inference Time', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Time (ms)')
        
        for bar, time_ms in zip(bars, inference_times):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{time_ms:.2f}ms', ha='center', va='bottom', fontweight='bold')
        
        # 4. Model Size Comparison
        ax4 = plt.subplot(2, 4, 4)
        model_sizes = [complexity_results[model]['model_size_mb'] for model in models]
        bars = ax4.bar(models, model_sizes, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax4.set_title('Model Size', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Size (MB)')
        
        for bar, size in zip(bars, model_sizes):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{size:.1f}MB', ha='center', va='bottom', fontweight='bold')
        
        # 5. Per-Class Accuracy Heatmap
        ax5 = plt.subplot(2, 2, 3)
        per_class_data = []
        for model in models:
            per_class_data.append([performance_results[model]['per_class_accuracy'][class_name] 
                                 for class_name in self.class_names])
        
        per_class_df = pd.DataFrame(per_class_data, index=models, columns=self.class_names)
        sns.heatmap(per_class_df, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=ax5, cbar_kws={'label': 'Accuracy (%)'})
        ax5.set_title('Per-Class Accuracy Heatmap', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Voice Commands')
        ax5.set_ylabel('Models')
        
        # 6. Performance vs Complexity Scatter Plot
        ax6 = plt.subplot(2, 2, 4)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        for i, model in enumerate(models):
            ax6.scatter(complexity_results[model]['total_parameters']/1e6, 
                       performance_results[model]['accuracy'], 
                       s=200, c=colors[i], alpha=0.7, label=model)
            ax6.annotate(model, 
                        (complexity_results[model]['total_parameters']/1e6, 
                         performance_results[model]['accuracy']),
                        xytext=(5, 5), textcoords='offset points', fontweight='bold')
        
        ax6.set_xlabel('Model Parameters (Millions)')
        ax6.set_ylabel('Accuracy (%)')
        ax6.set_title('Performance vs Complexity', fontsize=14, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../results/June11/model_comparison_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comparison_table(self, performance_results, complexity_results):
        """Create a detailed comparison table"""
        print("Creating comparison table...")
        
        # Combine training info if available
        table_data = []
        for model_name in self.models.keys():
            row = {
                'Model': model_name,
                'Accuracy (%)': f"{performance_results[model_name]['accuracy']:.2f}",
                'Parameters (M)': f"{complexity_results[model_name]['total_parameters']/1e6:.2f}",
                'Model Size (MB)': f"{complexity_results[model_name]['model_size_mb']:.1f}",
                'Inference Time (ms)': f"{performance_results[model_name]['avg_inference_time_ms']:.2f}",
                'Memory Usage (MB)': f"{complexity_results[model_name]['memory_usage_mb']:.1f}"
            }
            
            # Add training info if available
            if model_name in self.model_info:
                info = self.model_info[model_name]
                if 'total_training_time' in info:
                    row['Training Time (s)'] = f"{info['total_training_time']:.1f}"
                if 'best_val_accuracy' in info:
                    row['Best Val Acc (%)'] = f"{info['best_val_accuracy']:.2f}"
            
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        
        # Save to CSV
        df.to_csv('../results/June11/model_comparison_table.csv', index=False)
        
        # Display table
        print("\n" + "="*80)
        print("MODEL COMPARISON TABLE")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)
        
        return df
    
    def save_detailed_results(self, performance_results, complexity_results):
        """Save detailed results to JSON"""
        print("Saving detailed results...")
        
        detailed_results = {
            'performance': performance_results,
            'complexity': complexity_results,
            'model_info': self.model_info,
            'dataset_info': {
                'num_classes': self.num_classes,
                'class_names': self.class_names,
                'dataset_path': self.dataset_path
            }
        }
        
        # Remove numpy arrays for JSON serialization
        for model_name in performance_results:
            if 'predictions' in detailed_results['performance'][model_name]:
                del detailed_results['performance'][model_name]['predictions']
            if 'labels' in detailed_results['performance'][model_name]:
                del detailed_results['performance'][model_name]['labels']
        
        with open('../results/June11/detailed_model_comparison.json', 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print("✓ Detailed results saved to '../results/June11/detailed_model_comparison.json'")
    
    def run_complete_comparison(self):
        """Run the complete model comparison pipeline"""
        print("Starting complete model comparison...")
        print("="*60)
        
        # Load models and info
        self.load_models()
        self.load_training_info()
        
        if not self.models:
            print("No models loaded! Please train the models first.")
            return
        
        # Evaluate performance
        performance_results = self.evaluate_model_performance()
        
        # Measure complexity
        complexity_results = self.measure_model_complexity()
        
        # Create visualizations
        self.create_comparison_visualizations(performance_results, complexity_results)
        
        # Create comparison table
        comparison_df = self.create_comparison_table(performance_results, complexity_results)
        
        # Save detailed results
        self.save_detailed_results(performance_results, complexity_results)
        
        print("\n" + "="*60)
        print("MODEL COMPARISON COMPLETE!")
        print("="*60)
        print("Generated files:")
        print("- model_comparison_comprehensive.png")
        print("- model_comparison_table.csv")
        print("- detailed_model_comparison.json")
        print("="*60)
        
        return performance_results, complexity_results, comparison_df

# ------------------------ Main Function ------------------------
def main():
    print("Voice Command Model Comparison Tool")
    print("="*50)
    
    # Initialize comparison
    comparison = ModelComparison()
    
    # Run complete comparison
    results = comparison.run_complete_comparison()
    
    if results:
        performance_results, complexity_results, comparison_df = results
        
        # Print summary
        print("\nSUMMARY:")
        best_accuracy_model = max(performance_results.keys(), 
                                key=lambda x: performance_results[x]['accuracy'])
        fastest_model = min(performance_results.keys(), 
                          key=lambda x: performance_results[x]['avg_inference_time_ms'])
        smallest_model = min(complexity_results.keys(), 
                           key=lambda x: complexity_results[x]['total_parameters'])
        
        print(f"🏆 Best Accuracy: {best_accuracy_model} ({performance_results[best_accuracy_model]['accuracy']:.2f}%)")
        print(f"⚡ Fastest Inference: {fastest_model} ({performance_results[fastest_model]['avg_inference_time_ms']:.2f}ms)")
        print(f"📦 Smallest Model: {smallest_model} ({complexity_results[smallest_model]['total_parameters']/1e6:.2f}M params)")

if __name__ == "__main__":
    main()
