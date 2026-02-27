#!/usr/bin/env python3
"""
Audio-Visual Synchronization: Simple test script.

This script tests the basic functionality of the audio-visual synchronization system.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from src.utils.core import create_synthetic_dataset, set_seed, get_device
from src.models.sync_model import create_model
from src.features.extractors import AudioFeatureExtractor, VisualFeatureExtractor


def test_synthetic_data_creation():
    """Test synthetic dataset creation."""
    print("Testing synthetic data creation...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        create_synthetic_dataset(
            output_dir=tmp_dir,
            num_samples=5,
            duration=2.0,
            sample_rate=16000,
            fps=25
        )
        
        # Check if files were created
        audio_files = list(Path(tmp_dir).glob("*.wav"))
        video_files = list(Path(tmp_dir).glob("*.mp4"))
        
        assert len(audio_files) == 5, f"Expected 5 audio files, got {len(audio_files)}"
        assert len(video_files) == 5, f"Expected 5 video files, got {len(video_files)}"
        
        print("✅ Synthetic data creation test passed")


def test_feature_extractors():
    """Test feature extractors."""
    print("Testing feature extractors...")
    
    # Test audio feature extractor
    audio_extractor = AudioFeatureExtractor(
        sample_rate=16000,
        n_mfcc=13,
        n_mels=128
    )
    
    # Create dummy audio
    audio = torch.randn(16000)  # 1 second of audio
    audio_features = audio_extractor(audio)
    
    assert audio_features.shape[0] == 13, f"Expected 13 MFCC coefficients, got {audio_features.shape[0]}"
    assert audio_features.shape[1] > 0, "Expected non-zero time dimension"
    
    # Test visual feature extractor
    visual_extractor = VisualFeatureExtractor(
        input_size=(64, 64),
        feature_dim=128,
        backbone="simple_cnn"
    )
    
    # Create dummy video frames
    frames = torch.randn(25, 64, 64, 3)  # 25 frames, 64x64, RGB
    visual_features = visual_extractor(frames)
    
    assert visual_features.shape[0] == 25, f"Expected 25 frames, got {visual_features.shape[0]}"
    assert visual_features.shape[1] == 128, f"Expected 128 feature dim, got {visual_features.shape[1]}"
    
    print("✅ Feature extractors test passed")


def test_model_creation():
    """Test model creation."""
    print("Testing model creation...")
    
    # Create a minimal config
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
    
    config = Config()
    
    # Test cross-correlation model
    model = create_model(config)
    assert model.model_name == "cross_correlation", f"Expected cross_correlation model, got {model.model_name}"
    
    print("✅ Model creation test passed")


def test_device_selection():
    """Test device selection."""
    print("Testing device selection...")
    
    device = get_device("auto")
    assert isinstance(device, torch.device), f"Expected torch.device, got {type(device)}"
    
    print(f"✅ Device selection test passed - using device: {device}")


def test_seeding():
    """Test random seeding."""
    print("Testing random seeding...")
    
    set_seed(42)
    rand1 = torch.randn(10)
    
    set_seed(42)
    rand2 = torch.randn(10)
    
    assert torch.allclose(rand1, rand2), "Seeding not working properly"
    
    print("✅ Seeding test passed")


def main():
    """Run all tests."""
    print("Running Audio-Visual Synchronization Tests")
    print("=" * 50)
    
    try:
        test_seeding()
        test_device_selection()
        test_synthetic_data_creation()
        test_feature_extractors()
        test_model_creation()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed successfully!")
        print("The audio-visual synchronization system is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
