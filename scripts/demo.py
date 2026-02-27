#!/usr/bin/env python3
"""
Simple demo script for audio-visual synchronization.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from src.utils.core import create_synthetic_dataset, AudioVisualSyncDataset, set_seed, get_device
from src.models.sync_model import create_model
from src.metrics.sync_metrics import AudioVisualSyncEvaluator


def run_demo():
    """Run a simple demo of the audio-visual synchronization system."""
    print("Audio-Visual Synchronization Demo")
    print("=" * 50)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create synthetic data
    data_dir = "data/processed"
    os.makedirs(data_dir, exist_ok=True)
    
    print("Creating synthetic dataset...")
    create_synthetic_dataset(
        output_dir=data_dir,
        num_samples=10,
        duration=2.0,
        sample_rate=16000,
        fps=25
    )
    
    # Create minimal config
    class Config:
        def __init__(self):
            self.model = type('obj', (object,), {
                'name': 'cross_correlation',
                'audio_feature_dim': 13,
                'visual_feature_dim': 128,
                'hidden_dim': 512,
                'dropout': 0.1
            })()
            self.audio = type('obj', (object,), {
                'sample_rate': 16000,
                'n_mfcc': 13,
                'n_mels': 128,
                'n_fft': 2048,
                'hop_length': 512,
                'preemphasis': 0.97
            })()
            self.video = type('obj', (object,), {
                'target_fps': 25,
                'mouth_region_size': 64
            })()
            self.eval = type('obj', (object,), {
                'tolerance_frames': 2
            })()
            self.device = 'auto'
    
    config = Config()
    
    # Create dataset
    print("Loading dataset...")
    dataset = AudioVisualSyncDataset(data_dir, config, "train")
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Create model
    print("Creating model...")
    model = create_model(config)
    device = get_device(config.device)
    model = model.to(device)
    model.eval()
    
    print(f"Using device: {device}")
    print(f"Model type: {model.model_name}")
    
    # Test on a few samples
    print("\nTesting synchronization on samples...")
    evaluator = AudioVisualSyncEvaluator(tolerance_frames=2)
    
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        
        # Add batch dimension
        audio = sample['audio'].unsqueeze(0).to(device)
        video_frames = sample['video_frames'].unsqueeze(0).to(device)
        mouth_regions = sample['mouth_regions'].unsqueeze(0).to(device)
        
        # Run model
        with torch.no_grad():
            outputs = model(audio, video_frames, mouth_regions)
        
        lag = outputs['lag'].item()
        confidence = outputs['correlation_score'].item()
        
        print(f"Sample {i+1}:")
        print(f"  Lag: {lag} frames")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Sample ID: {sample['sample_id']}")
        
        # Update evaluator (assuming ground truth lag is 0)
        evaluator.sync_metrics.update(lag, 0, confidence)
    
    # Print evaluation summary
    print("\nEvaluation Summary:")
    print(evaluator.get_summary())
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
