"""
Edge AI Deployment Module
Export models to TorchScript for edge device inference
Includes quantization and optimization for resource-constrained devices
"""

import torch
import torch.nn as nn
import numpy as np
import time
import os
from typing import Dict, Any


class EdgeAIOptimizer:
    """
    Optimize and export models for edge deployment.
    Supports TorchScript, quantization, and pruning.
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def export_to_torchscript(self, save_path, example_input, optimize=True):
        """
        Export model to TorchScript format.
        
        Args:
            save_path: Path to save TorchScript model
            example_input: Example input tensor for tracing
            optimize: Apply TorchScript optimization
        
        Returns:
            TorchScript model
        """
        print(f"\nExporting model to TorchScript...")
        
        # Set model to eval mode
        self.model.eval()
        
        # Trace the model
        with torch.no_grad():
            traced_model = torch.jit.trace(self.model, example_input)
        
        # Optimize if requested
        if optimize:
            traced_model = torch.jit.optimize_for_inference(traced_model)
            print("  Applied TorchScript optimization for inference")
        
        # Save model
        traced_model.save(save_path)
        
        # Get file size
        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        
        print(f"  TorchScript model saved: {save_path}")
        print(f"  Model size: {file_size_mb:.2f} MB")
        
        return traced_model
    
    def quantize_model(self, save_path, calibration_loader=None):
        """
        Apply dynamic quantization to reduce model size.
        
        Args:
            save_path: Path to save quantized model
            calibration_loader: Data loader for calibration (optional)
        
        Returns:
            Quantized model
        """
        print(f"\nApplying dynamic quantization...")
        
        # Dynamic quantization (post-training)
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.LSTM, nn.Linear},  # Quantize LSTM and Linear layers
            dtype=torch.qint8
        )
        
        # Save quantized model
        torch.save(quantized_model.state_dict(), save_path)
        
        # Get file size
        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        
        print(f"  Quantized model saved: {save_path}")
        print(f"  Model size: {file_size_mb:.2f} MB")
        
        return quantized_model
    
    def benchmark_inference(self, example_input, n_runs=100):
        """
        Benchmark inference latency and throughput.
        
        Args:
            example_input: Example input tensor
            n_runs: Number of inference runs
        
        Returns:
            Benchmark metrics
        """
        print(f"\nBenchmarking inference performance...")
        
        self.model.eval()
        
        # Warm-up
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(example_input)
        
        # Benchmark
        latencies = []
        
        with torch.no_grad():
            for _ in range(n_runs):
                start_time = time.time()
                _ = self.model(example_input)
                latency = (time.time() - start_time) * 1000  # Convert to ms
                latencies.append(latency)
        
        metrics = {
            'mean_latency_ms': np.mean(latencies),
            'std_latency_ms': np.std(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'throughput_per_sec': 1000 / np.mean(latencies)
        }
        
        print(f"  Mean latency: {metrics['mean_latency_ms']:.2f} ± {metrics['std_latency_ms']:.2f} ms")
        print(f"  Min latency: {metrics['min_latency_ms']:.2f} ms")
        print(f"  Max latency: {metrics['max_latency_ms']:.2f} ms")
        print(f"  Throughput: {metrics['throughput_per_sec']:.2f} inferences/sec")
        
        return metrics
    
    def compare_original_vs_optimized(self, original_model, optimized_model, 
                                     example_input, n_samples=100):
        """
        Compare original model with optimized version.
        
        Args:
            original_model: Original PyTorch model
            optimized_model: Optimized model (TorchScript/quantized)
            example_input: Example input
            n_samples: Number of samples to compare
        
        Returns:
            Comparison metrics
        """
        print(f"\nComparing original vs optimized model...")
        
        original_model.eval()
        
        # Check if optimized is TorchScript
        is_torchscript = isinstance(optimized_model, torch.jit.ScriptModule)
        
        original_outputs = []
        optimized_outputs = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                # Original prediction
                orig_out = original_model(example_input)
                if isinstance(orig_out, tuple):
                    orig_out = orig_out[0]  # Energy prediction
                original_outputs.append(orig_out.cpu().numpy())
                
                # Optimized prediction
                if is_torchscript:
                    opt_out = optimized_model(example_input)
                else:
                    opt_out = optimized_model(example_input)
                    if isinstance(opt_out, tuple):
                        opt_out = opt_out[0]
                optimized_outputs.append(opt_out.cpu().numpy())
        
        original_outputs = np.array(original_outputs)
        optimized_outputs = np.array(optimized_outputs)
        
        # Calculate difference
        abs_diff = np.abs(original_outputs - optimized_outputs)
        rel_diff = abs_diff / (np.abs(original_outputs) + 1e-8)
        
        metrics = {
            'mean_abs_diff': np.mean(abs_diff),
            'max_abs_diff': np.max(abs_diff),
            'mean_rel_diff': np.mean(rel_diff) * 100,  # Percentage
            'max_rel_diff': np.max(rel_diff) * 100
        }
        
        print(f"  Mean absolute difference: {metrics['mean_abs_diff']:.6f}")
        print(f"  Max absolute difference: {metrics['max_abs_diff']:.6f}")
        print(f"  Mean relative difference: {metrics['mean_rel_diff']:.4f}%")
        print(f"  Max relative difference: {metrics['max_rel_diff']:.4f}%")
        
        return metrics


class EdgeAIInferenceEngine:
    """
    Lightweight inference engine for edge devices.
    Loads TorchScript models and performs real-time inference.
    """
    
    def __init__(self, model_path, device='cpu'):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to TorchScript model
            device: Device for inference (cpu/cuda)
        """
        self.device = device
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()
        
        print(f"Edge AI Inference Engine initialized")
        print(f"  Model loaded from: {model_path}")
        print(f"  Device: {device}")
    
    def predict(self, state):
        """
        Perform inference on input state.
        
        Args:
            state: Input state (numpy array or tensor)
        
        Returns:
            Prediction
        """
        # Convert to tensor if needed
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        
        # Add batch dimension if needed
        if len(state.shape) == 2:
            state = state.unsqueeze(0)
        
        state = state.to(self.device)
        
        with torch.no_grad():
            prediction = self.model(state)
        
        # Handle tuple output
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        
        return prediction.cpu().numpy()
    
    def predict_batch(self, states):
        """
        Batch inference for multiple states.
        
        Args:
            states: Batch of states (numpy array or tensor)
        
        Returns:
            Batch of predictions
        """
        if isinstance(states, np.ndarray):
            states = torch.FloatTensor(states)
        
        states = states.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(states)
        
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        return predictions.cpu().numpy()


def create_edge_deployment_package(models_dict, example_inputs_dict, 
                                   output_dir='../models/edge_deployment'):
    """
    Create complete edge deployment package with all models.
    
    Args:
        models_dict: Dictionary of {model_name: model}
        example_inputs_dict: Dictionary of {model_name: example_input}
        output_dir: Output directory for deployment package
    
    Returns:
        Deployment package metadata
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("Creating Edge Deployment Package")
    print("="*80)
    
    deployment_info = {}
    
    for model_name, model in models_dict.items():
        print(f"\nProcessing {model_name}...")
        
        example_input = example_inputs_dict[model_name]
        
        # Create optimizer
        optimizer = EdgeAIOptimizer(model)
        
        # Export to TorchScript
        torchscript_path = os.path.join(output_dir, f"{model_name}_torchscript.pt")
        traced_model = optimizer.export_to_torchscript(torchscript_path, example_input)
        
        # Quantize model
        quantized_path = os.path.join(output_dir, f"{model_name}_quantized.pth")
        quantized_model = optimizer.quantize_model(quantized_path)
        
        # Benchmark
        original_metrics = optimizer.benchmark_inference(example_input)
        
        # Benchmark TorchScript
        print(f"\nBenchmarking TorchScript version...")
        ts_optimizer = EdgeAIOptimizer(traced_model)
        ts_metrics = ts_optimizer.benchmark_inference(example_input)
        
        # Compare accuracy
        comparison = optimizer.compare_original_vs_optimized(
            model, traced_model, example_input
        )
        
        # Store info
        deployment_info[model_name] = {
            'torchscript_path': torchscript_path,
            'quantized_path': quantized_path,
            'original_latency_ms': original_metrics['mean_latency_ms'],
            'torchscript_latency_ms': ts_metrics['mean_latency_ms'],
            'speedup': original_metrics['mean_latency_ms'] / ts_metrics['mean_latency_ms'],
            'accuracy_diff': comparison['mean_abs_diff'],
            'model_size_mb': os.path.getsize(torchscript_path) / (1024 * 1024)
        }
    
    # Save deployment metadata
    import json
    metadata_path = os.path.join(output_dir, 'deployment_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    print("\n" + "="*80)
    print("Edge Deployment Package Created!")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # Print summary
    print("\nDeployment Summary:")
    for model_name, info in deployment_info.items():
        print(f"\n{model_name}:")
        print(f"  Latency: {info['original_latency_ms']:.2f} → {info['torchscript_latency_ms']:.2f} ms")
        print(f"  Speedup: {info['speedup']:.2f}x")
        print(f"  Accuracy difference: {info['accuracy_diff']:.6f}")
        print(f"  Model size: {info['model_size_mb']:.2f} MB")
    
    return deployment_info


def generate_edge_deployment_report(deployment_info, save_path='../figures'):
    """
    Generate visualization report for edge deployment.
    
    Args:
        deployment_info: Deployment metadata dictionary
        save_path: Path to save figures
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    os.makedirs(save_path, exist_ok=True)
    
    # Extract data
    model_names = list(deployment_info.keys())
    original_latencies = [info['original_latency_ms'] for info in deployment_info.values()]
    optimized_latencies = [info['torchscript_latency_ms'] for info in deployment_info.values()]
    speedups = [info['speedup'] for info in deployment_info.values()]
    model_sizes = [info['model_size_mb'] for info in deployment_info.values()]
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Latency comparison
    x = np.arange(len(model_names))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, original_latencies, width, label='Original', 
                   color='#2E86AB', alpha=0.7)
    axes[0, 0].bar(x + width/2, optimized_latencies, width, label='TorchScript', 
                   color='#F18F01', alpha=0.7)
    axes[0, 0].set_xlabel('Model', fontweight='bold', fontsize=12)
    axes[0, 0].set_ylabel('Latency (ms)', fontweight='bold', fontsize=12)
    axes[0, 0].set_title('Inference Latency Comparison', fontweight='bold', fontsize=14)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Speedup
    axes[0, 1].bar(model_names, speedups, color='#A23B72', alpha=0.7)
    axes[0, 1].axhline(y=1.0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Model', fontweight='bold', fontsize=12)
    axes[0, 1].set_ylabel('Speedup Factor', fontweight='bold', fontsize=12)
    axes[0, 1].set_title('TorchScript Optimization Speedup', fontweight='bold', fontsize=14)
    axes[0, 1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Model size
    axes[1, 0].bar(model_names, model_sizes, color='#FF6B6B', alpha=0.7)
    axes[1, 0].set_xlabel('Model', fontweight='bold', fontsize=12)
    axes[1, 0].set_ylabel('Model Size (MB)', fontweight='bold', fontsize=12)
    axes[1, 0].set_title('Deployed Model Size', fontweight='bold', fontsize=14)
    axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Summary metrics table
    axes[1, 1].axis('off')
    
    summary_text = "Edge Deployment Summary\n" + "="*40 + "\n\n"
    for model_name, info in deployment_info.items():
        summary_text += f"{model_name}:\n"
        summary_text += f"  Latency: {info['torchscript_latency_ms']:.2f} ms\n"
        summary_text += f"  Speedup: {info['speedup']:.2f}x\n"
        summary_text += f"  Size: {info['model_size_mb']:.2f} MB\n\n"
    
    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                   verticalalignment='top', fontfamily='monospace', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Edge AI Deployment Performance Analysis', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filename = os.path.join(save_path, 'edge_deployment_analysis.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Edge deployment report saved: {filename}")
    plt.close()


if __name__ == "__main__":
    print("="*80)
    print("Testing Edge AI Deployment")
    print("="*80)
    
    from deep_learning_models import LSTMEnergyPredictor
    
    # Create a test model
    print("\nCreating test LSTM model...")
    model = LSTMEnergyPredictor(input_size=15, hidden_size=64, num_layers=2)
    
    # Create example input
    batch_size = 1
    seq_length = 24
    n_features = 15
    example_input = torch.randn(batch_size, seq_length, n_features)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test optimizer
    print("\nTesting EdgeAIOptimizer...")
    optimizer = EdgeAIOptimizer(model)
    
    # Export to TorchScript
    torchscript_path = '../models/test_torchscript.pt'
    os.makedirs('../models', exist_ok=True)
    traced_model = optimizer.export_to_torchscript(torchscript_path, example_input)
    
    # Benchmark
    metrics = optimizer.benchmark_inference(example_input)
    
    # Test inference engine
    print("\nTesting EdgeAIInferenceEngine...")
    engine = EdgeAIInferenceEngine(torchscript_path)
    prediction = engine.predict(example_input.numpy())
    print(f"  Prediction shape: {prediction.shape}")
    
    print("\n" + "="*80)
    print("Edge AI Deployment Test Complete!")
    print("="*80)
