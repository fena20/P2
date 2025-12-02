"""
Edge AI Module: TorchScript Export for Local Inference
Converts trained models to TorchScript for deployment on edge devices
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Optional

class EdgeAIExporter:
    """
    Exports PyTorch models to TorchScript for edge deployment.
    TorchScript enables efficient inference on edge devices without Python runtime.
    """
    
    def __init__(self, model: nn.Module, device='cpu'):
        """
        Initialize exporter.
        
        Args:
            model: Trained PyTorch model
            device: Target device for export ('cpu' or 'cuda')
        """
        self.model = model
        self.device = device
        self.model.eval()  # Set to evaluation mode
    
    def export_torchscript(self, example_input: torch.Tensor,
                          save_path: str, optimize=True):
        """
        Export model to TorchScript format.
        
        Args:
            example_input: Example input tensor for tracing
            save_path: Path to save the exported model
            optimize: Whether to apply optimizations
        
        Returns:
            traced_model: Traced TorchScript model
        """
        print(f"Exporting model to TorchScript...")
        print(f"Example input shape: {example_input.shape}")
        
        # Move model and input to target device
        self.model = self.model.to(self.device)
        example_input = example_input.to(self.device)
        
        # Trace the model
        try:
            traced_model = torch.jit.trace(self.model, example_input)
            
            if optimize:
                # Apply optimizations
                traced_model = torch.jit.optimize_for_inference(traced_model)
            
            # Save traced model
            traced_model.save(save_path)
            print(f"Model exported successfully to: {save_path}")
            
            # Verify export
            self.verify_export(traced_model, example_input)
            
            return traced_model
        
        except Exception as e:
            print(f"Error during export: {e}")
            # Try script mode if tracing fails
            print("Attempting script mode...")
            try:
                scripted_model = torch.jit.script(self.model)
                scripted_model.save(save_path)
                print(f"Model exported using script mode to: {save_path}")
                return scripted_model
            except Exception as e2:
                print(f"Script mode also failed: {e2}")
                raise
    
    def verify_export(self, traced_model: torch.jit.ScriptModule,
                     example_input: torch.Tensor):
        """
        Verify that exported model produces same output as original.
        """
        print("Verifying exported model...")
        
        with torch.no_grad():
            # Original model output
            original_output = self.model(example_input)
            
            # Traced model output
            traced_output = traced_model(example_input)
            
            # Compare outputs
            if isinstance(original_output, tuple):
                # Multi-output model
                for i, (orig, traced) in enumerate(zip(original_output, traced_output)):
                    max_diff = torch.max(torch.abs(orig - traced)).item()
                    print(f"Output {i} max difference: {max_diff:.6f}")
                    if max_diff > 1e-5:
                        print(f"Warning: Significant difference detected!")
            else:
                max_diff = torch.max(torch.abs(original_output - traced_output)).item()
                print(f"Max difference: {max_diff:.6f}")
                if max_diff > 1e-5:
                    print(f"Warning: Significant difference detected!")
        
        print("Verification complete.")
    
    def export_quantized(self, example_input: torch.Tensor, save_path: str):
        """
        Export quantized model for even smaller size and faster inference.
        
        Args:
            example_input: Example input tensor
            save_path: Path to save quantized model
        """
        print("Exporting quantized model...")
        
        # Move to CPU (quantization typically done on CPU)
        self.model = self.model.cpu()
        example_input = example_input.cpu()
        
        # Quantize model
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear, nn.LSTM},  # Layers to quantize
            dtype=torch.qint8
        )
        
        # Trace quantized model
        traced_quantized = torch.jit.trace(quantized_model, example_input)
        traced_quantized.save(save_path)
        
        print(f"Quantized model exported to: {save_path}")
        
        # Compare model sizes
        original_size = Path(save_path.replace('_quantized', '')).stat().st_size if Path(save_path.replace('_quantized', '')).exists() else 0
        quantized_size = Path(save_path).stat().st_size
        if original_size > 0:
            compression_ratio = original_size / quantized_size
            print(f"Compression ratio: {compression_ratio:.2f}x")
        
        return traced_quantized

class EdgeInferenceEngine:
    """
    Inference engine for edge devices using TorchScript models.
    """
    
    def __init__(self, model_path: str, device='cpu'):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to TorchScript model
            device: Device to run inference on
        """
        self.device = device
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()
        print(f"Loaded TorchScript model from: {model_path}")
        print(f"Running on device: {device}")
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference on input data.
        
        Args:
            input_data: Input array (can be single sample or batch)
        
        Returns:
            predictions: Model predictions
        """
        # Convert to tensor
        if isinstance(input_data, np.ndarray):
            input_tensor = torch.FloatTensor(input_data)
        else:
            input_tensor = input_data
        
        # Add batch dimension if needed
        if len(input_tensor.shape) == 2:
            input_tensor = input_tensor.unsqueeze(0)
        
        input_tensor = input_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)
            
            # Handle tuple outputs
            if isinstance(output, tuple):
                output = output[0]  # Take first output (energy prediction)
            
            # Convert back to numpy
            predictions = output.cpu().numpy()
            
            # Remove batch dimension if single sample
            if predictions.shape[0] == 1 and len(predictions.shape) > 1:
                predictions = predictions.squeeze(0)
        
        return predictions
    
    def benchmark_inference(self, input_data: np.ndarray, num_runs=100):
        """
        Benchmark inference speed.
        
        Args:
            input_data: Example input data
            num_runs: Number of inference runs for benchmarking
        
        Returns:
            avg_time: Average inference time in milliseconds
        """
        import time
        
        # Warmup
        for _ in range(10):
            _ = self.predict(input_data)
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.time()
            _ = self.predict(input_data)
            end = time.time()
            times.append((end - start) * 1000)  # Convert to ms
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"Inference Benchmark ({num_runs} runs):")
        print(f"Average time: {avg_time:.2f} ms")
        print(f"Std deviation: {std_time:.2f} ms")
        print(f"Min time: {np.min(times):.2f} ms")
        print(f"Max time: {np.max(times):.2f} ms")
        
        return avg_time

def export_model_for_edge(model: nn.Module, example_input: torch.Tensor,
                         output_dir: str = 'models/edge_ai',
                         model_name: str = 'edge_model'):
    """
    Convenience function to export model for edge deployment.
    
    Args:
        model: Trained PyTorch model
        example_input: Example input tensor
        output_dir: Output directory
        model_name: Base name for exported models
    
    Returns:
        Dict with paths to exported models
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    exporter = EdgeAIExporter(model, device='cpu')
    
    # Export standard TorchScript
    torchscript_path = f"{output_dir}/{model_name}.pt"
    traced_model = exporter.export_torchscript(
        example_input, torchscript_path, optimize=True
    )
    
    # Export quantized version
    quantized_path = f"{output_dir}/{model_name}_quantized.pt"
    try:
        quantized_model = exporter.export_quantized(example_input, quantized_path)
    except Exception as e:
        print(f"Quantization failed: {e}")
        quantized_model = None
    
    return {
        'torchscript': torchscript_path,
        'quantized': quantized_path if quantized_model else None,
        'model': traced_model
    }
