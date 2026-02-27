"""
Audio-Visual Synchronization: Core utilities and device management.

This module provides utilities for device management, seeding, and common
audio-visual processing functions.
"""

import os
import random
from typing import Optional, Union, Tuple, Any
import warnings

import numpy as np
import torch
import torchaudio
import librosa
import cv2
from omegaconf import DictConfig


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: str = "auto") -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        device: Device preference ("auto", "cuda", "mps", "cpu").
        
    Returns:
        PyTorch device object.
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    return torch.device(device)


def load_audio(
    file_path: str, 
    sample_rate: Optional[int] = None,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, int]:
    """Load audio file with proper device handling.
    
    Args:
        file_path: Path to audio file.
        sample_rate: Target sample rate (None to keep original).
        device: Device to load tensor on.
        
    Returns:
        Tuple of (audio_tensor, sample_rate).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    # Load with librosa for better format support
    audio, sr = librosa.load(file_path, sr=sample_rate)
    
    # Convert to tensor
    audio_tensor = torch.from_numpy(audio).float()
    
    if device is not None:
        audio_tensor = audio_tensor.to(device)
    
    return audio_tensor, sr


def load_video_frames(
    video_path: str,
    target_fps: Optional[int] = None,
    max_frames: Optional[int] = None
) -> Tuple[np.ndarray, float]:
    """Load video frames with optional frame rate conversion.
    
    Args:
        video_path: Path to video file.
        target_fps: Target frame rate (None to keep original).
        max_frames: Maximum number of frames to load.
        
    Returns:
        Tuple of (frames_array, actual_fps).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine frame sampling
    if target_fps is not None and target_fps != original_fps:
        frame_skip = int(original_fps / target_fps)
    else:
        frame_skip = 1
    
    frames = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_skip == 0:
            frames.append(frame)
            
        frame_count += 1
        
        if max_frames is not None and len(frames) >= max_frames:
            break
    
    cap.release()
    
    if not frames:
        raise ValueError("No frames could be loaded from video")
    
    actual_fps = original_fps / frame_skip
    return np.array(frames), actual_fps


def extract_mouth_region(
    frame: np.ndarray,
    cascade_path: Optional[str] = None,
    target_size: Tuple[int, int] = (64, 64)
) -> Optional[np.ndarray]:
    """Extract mouth region from a face frame.
    
    Args:
        frame: Input frame (BGR format).
        cascade_path: Path to mouth cascade classifier.
        target_size: Target size for mouth region.
        
    Returns:
        Extracted mouth region or None if not found.
    """
    if cascade_path is None:
        cascade_path = cv2.data.haarcascades + 'haarcascade_mcs_mouth.xml'
    
    if not os.path.exists(cascade_path):
        warnings.warn(f"Mouth cascade not found at {cascade_path}, using full frame")
        return cv2.resize(frame, target_size)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mouth_cascade = cv2.CascadeClassifier(cascade_path)
    
    mouths = mouth_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30)
    )
    
    if len(mouths) > 0:
        # Use the largest detected mouth
        mouth = max(mouths, key=lambda x: x[2] * x[3])
        x, y, w, h = mouth
        
        # Extract mouth region (focus on lower half)
        mouth_roi = frame[y + h // 2:y + h, x:x + w]
        return cv2.resize(mouth_roi, target_size)
    
    # Fallback: use center region of frame
    h, w = frame.shape[:2]
    center_region = frame[h//3:2*h//3, w//3:2*w//3]
    return cv2.resize(center_region, target_size)


def normalize_features(features: np.ndarray) -> np.ndarray:
    """Normalize features to zero mean and unit variance.
    
    Args:
        features: Input features array.
        
    Returns:
        Normalized features.
    """
    if features.std() == 0:
        return features - features.mean()
    
    return (features - features.mean()) / features.std()


def create_synthetic_dataset(
    output_dir: str,
    num_samples: int = 100,
    duration: float = 3.0,
    sample_rate: int = 16000,
    fps: int = 25
) -> None:
    """Create a synthetic dataset for testing audio-visual synchronization.
    
    Args:
        output_dir: Directory to save synthetic data.
        num_samples: Number of samples to generate.
        duration: Duration of each sample in seconds.
        sample_rate: Audio sample rate.
        fps: Video frame rate.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_samples):
        # Generate synthetic audio (sine wave with varying frequency)
        t = np.linspace(0, duration, int(sample_rate * duration))
        freq = 440 + 100 * np.sin(2 * np.pi * t / duration)
        audio = np.sin(2 * np.pi * freq * t) * 0.5
        
        # Generate synthetic video frames (moving circle representing mouth movement)
        num_frames = int(duration * fps)
        frames = []
        
        for frame_idx in range(num_frames):
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
            
            # Moving circle representing mouth movement
            center_x = 112 + 50 * np.sin(2 * np.pi * frame_idx / num_frames)
            center_y = 150 + 20 * np.cos(2 * np.pi * frame_idx / num_frames)
            
            cv2.circle(frame, (int(center_x), int(center_y)), 20, (255, 255, 255), -1)
            frames.append(frame)
        
        # Save audio
        audio_path = os.path.join(output_dir, f"sample_{i:03d}.wav")
        import soundfile as sf
        sf.write(audio_path, audio, sample_rate)
        
        # Save video
        video_path = os.path.join(output_dir, f"sample_{i:03d}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (224, 224))
        
        for frame in frames:
            out.write(frame)
        out.release()
        
        # Create metadata
        metadata = {
            "audio_path": audio_path,
            "video_path": video_path,
            "duration": duration,
            "sample_rate": sample_rate,
            "fps": fps,
            "true_lag": 0  # Synchronized by default
        }
        
        # Save metadata as simple text file for now
        metadata_path = os.path.join(output_dir, f"sample_{i:03d}.txt")
        with open(metadata_path, 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")


class AudioVisualSyncDataset:
    """Dataset class for audio-visual synchronization tasks."""
    
    def __init__(
        self,
        data_dir: str,
        config: DictConfig,
        split: str = "train"
    ):
        """Initialize dataset.
        
        Args:
            data_dir: Directory containing data files.
            config: Configuration object.
            split: Dataset split ("train", "val", "test").
        """
        self.data_dir = data_dir
        self.config = config
        self.split = split
        self.samples = self._load_samples()
    
    def _load_samples(self) -> list:
        """Load sample metadata."""
        samples = []
        
        # Look for audio files and corresponding video files
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.wav'):
                audio_path = os.path.join(self.data_dir, filename)
                video_path = audio_path.replace('.wav', '.mp4')
                
                if os.path.exists(video_path):
                    samples.append({
                        'audio_path': audio_path,
                        'video_path': video_path,
                        'id': filename.replace('.wav', '')
                    })
        
        return samples
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        """Get a sample from the dataset."""
        sample = self.samples[idx]
        
        # Load audio
        audio, sr = load_audio(
            sample['audio_path'],
            sample_rate=self.config.audio.sample_rate
        )
        
        # Load video frames
        frames, fps = load_video_frames(
            sample['video_path'],
            target_fps=self.config.video.target_fps
        )
        
        # Extract mouth regions
        mouth_regions = []
        for frame in frames:
            mouth_region = extract_mouth_region(
                frame,
                target_size=(self.config.video.mouth_region_size, 
                           self.config.video.mouth_region_size)
            )
            mouth_regions.append(mouth_region)
        
        mouth_regions = np.array(mouth_regions)
        
        return {
            'audio': audio,
            'video_frames': torch.from_numpy(frames).float(),
            'mouth_regions': torch.from_numpy(mouth_regions).float(),
            'sample_id': sample['id'],
            'audio_sr': sr,
            'video_fps': fps
        }
