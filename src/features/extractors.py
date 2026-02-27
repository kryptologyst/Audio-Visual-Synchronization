"""
Audio-Visual Synchronization: Feature extraction modules.

This module provides audio and visual feature extraction capabilities
for audio-visual synchronization tasks.
"""

from typing import Tuple, Optional, Union
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import cv2
from scipy.signal import correlate


class AudioFeatureExtractor(nn.Module):
    """Audio feature extractor using MFCC and mel-spectrogram features."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        preemphasis: float = 0.97
    ):
        """Initialize audio feature extractor.
        
        Args:
            sample_rate: Audio sample rate.
            n_mfcc: Number of MFCC coefficients.
            n_mels: Number of mel filter banks.
            n_fft: FFT window size.
            hop_length: Hop length for STFT.
            preemphasis: Preemphasis coefficient.
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.preemphasis = preemphasis
        
        # Mel filter bank
        self.mel_filters = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels
        )
        self.register_buffer('mel_filter_bank', torch.from_numpy(self.mel_filters).float())
        
        # DCT matrix for MFCC
        self.dct_matrix = self._create_dct_matrix(n_mels, n_mfcc)
        self.register_buffer('dct_mat', torch.from_numpy(self.dct_matrix).float())
    
    def _create_dct_matrix(self, n_mels: int, n_mfcc: int) -> np.ndarray:
        """Create DCT matrix for MFCC computation."""
        dct_matrix = np.zeros((n_mfcc, n_mels))
        for i in range(n_mfcc):
            for j in range(n_mels):
                dct_matrix[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * n_mels))
        return dct_matrix
    
    def preemphasis_filter(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply preemphasis filter to audio."""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Apply preemphasis: y[n] = x[n] - preemphasis * x[n-1]
        emphasized = torch.zeros_like(audio)
        emphasized[:, 0] = audio[:, 0]
        emphasized[:, 1:] = audio[:, 1:] - self.preemphasis * audio[:, :-1]
        
        return emphasized.squeeze(0) if audio.dim() == 2 else emphasized
    
    def extract_mel_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract mel-spectrogram from audio."""
        # Apply preemphasis
        audio = self.preemphasis_filter(audio)
        
        # Compute STFT
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft, device=audio.device),
            return_complex=True
        )
        
        # Compute power spectrogram
        power_spec = torch.abs(stft) ** 2
        
        # Apply mel filter bank
        mel_spec = torch.matmul(self.mel_filter_bank, power_spec)
        
        # Convert to log scale
        log_mel_spec = torch.log(mel_spec + 1e-8)
        
        return log_mel_spec
    
    def extract_mfcc(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract MFCC features from audio."""
        # Get mel-spectrogram
        mel_spec = self.extract_mel_spectrogram(audio)
        
        # Apply DCT to get MFCC
        mfcc = torch.matmul(self.dct_mat, mel_spec)
        
        return mfcc
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract audio features.
        
        Args:
            audio: Input audio tensor of shape (T,) or (B, T).
            
        Returns:
            MFCC features of shape (n_mfcc, T') or (B, n_mfcc, T').
        """
        return self.extract_mfcc(audio)


class VisualFeatureExtractor(nn.Module):
    """Visual feature extractor for mouth region analysis."""
    
    def __init__(
        self,
        input_size: Tuple[int, int] = (64, 64),
        feature_dim: int = 128,
        backbone: str = "simple_cnn"
    ):
        """Initialize visual feature extractor.
        
        Args:
            input_size: Input image size (height, width).
            feature_dim: Output feature dimension.
            backbone: Backbone architecture ("simple_cnn", "resnet").
        """
        super().__init__()
        self.input_size = input_size
        self.feature_dim = feature_dim
        self.backbone = backbone
        
        if backbone == "simple_cnn":
            self.feature_extractor = self._build_simple_cnn()
        elif backbone == "resnet":
            self.feature_extractor = self._build_resnet()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
    
    def _build_simple_cnn(self) -> nn.Module:
        """Build simple CNN backbone."""
        return nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            
            # Final projection
            nn.Linear(128, self.feature_dim)
        )
    
    def _build_resnet(self) -> nn.Module:
        """Build ResNet-based backbone."""
        # Use a pre-trained ResNet18 as backbone
        import torchvision.models as models
        
        resnet = models.resnet18(pretrained=True)
        # Remove the final classification layer
        resnet = nn.Sequential(*list(resnet.children())[:-1])
        
        # Add projection layer
        projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, self.feature_dim)
        )
        
        return nn.Sequential(resnet, projection)
    
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Extract visual features from video frames.
        
        Args:
            frames: Input frames of shape (T, H, W, C) or (B, T, H, W, C).
            
        Returns:
            Visual features of shape (T, feature_dim) or (B, T, feature_dim).
        """
        batch_dim = frames.dim() == 5
        
        if batch_dim:
            B, T, H, W, C = frames.shape
            frames = frames.view(B * T, C, H, W)
        else:
            T, H, W, C = frames.shape
            frames = frames.view(T, C, H, W)
        
        # Extract features
        features = self.feature_extractor(frames)
        
        if batch_dim:
            features = features.view(B, T, self.feature_dim)
        else:
            features = features.view(T, self.feature_dim)
        
        return features


class CrossCorrelationSync:
    """Cross-correlation based audio-visual synchronization."""
    
    def __init__(self, normalize: bool = True):
        """Initialize cross-correlation synchronizer.
        
        Args:
            normalize: Whether to normalize features before correlation.
        """
        self.normalize = normalize
    
    def synchronize(
        self,
        audio_features: np.ndarray,
        visual_features: np.ndarray
    ) -> Tuple[int, float]:
        """Synchronize audio and visual features using cross-correlation.
        
        Args:
            audio_features: Audio features of shape (D_audio, T_audio) or (T_audio, D_audio).
            visual_features: Visual features of shape (T_visual, D_visual).
            
        Returns:
            Tuple of (optimal_lag, correlation_score).
        """
        # Ensure features are 2D
        if audio_features.ndim == 1:
            audio_features = audio_features.reshape(-1, 1)
        if visual_features.ndim == 1:
            visual_features = visual_features.reshape(-1, 1)
        
        # Handle extra batch dimensions
        while visual_features.ndim > 2:
            visual_features = visual_features.squeeze(0)
        
        # Handle different audio feature formats
        # If audio features have more time steps than feature dimensions, transpose
        if audio_features.shape[0] < audio_features.shape[1]:
            audio_features = audio_features.T
        
        # Normalize features if requested
        if self.normalize:
            audio_features = self._normalize_features(audio_features)
            visual_features = self._normalize_features(visual_features)
        
        # Compute cross-correlation for each feature dimension
        correlations = []
        min_length = min(audio_features.shape[0], visual_features.shape[0])
        min_feature_dim = min(audio_features.shape[1], visual_features.shape[1])
        
        for d in range(min_feature_dim):
            # Truncate to same length for correlation
            audio_dim = audio_features[:min_length, d]
            visual_dim = visual_features[:min_length, d]
            
            corr = correlate(audio_dim, visual_dim, mode='full')
            correlations.append(corr)
        
        # Average correlations across dimensions
        avg_correlation = np.mean(correlations, axis=0)
        
        # Find optimal lag
        optimal_lag_idx = np.argmax(avg_correlation)
        optimal_lag = optimal_lag_idx - min_length + 1
        max_correlation = avg_correlation[optimal_lag_idx]
        
        return optimal_lag, max_correlation
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to zero mean and unit variance."""
        normalized = np.zeros_like(features)
        for d in range(features.shape[1]):
            feature_dim = features[:, d]
            if feature_dim.std() > 0:
                normalized[:, d] = (feature_dim - feature_dim.mean()) / feature_dim.std()
            else:
                normalized[:, d] = feature_dim - feature_dim.mean()
        
        return normalized


class NeuralSyncModel(nn.Module):
    """Neural network-based audio-visual synchronization model."""
    
    def __init__(
        self,
        audio_feature_dim: int = 13,
        visual_feature_dim: int = 128,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """Initialize neural synchronization model.
        
        Args:
            audio_feature_dim: Audio feature dimension.
            visual_feature_dim: Visual feature dimension.
            hidden_dim: Hidden layer dimension.
            num_layers: Number of LSTM layers.
            dropout: Dropout rate.
        """
        super().__init__()
        self.audio_feature_dim = audio_feature_dim
        self.visual_feature_dim = visual_feature_dim
        self.hidden_dim = hidden_dim
        
        # Audio encoder
        self.audio_encoder = nn.LSTM(
            input_size=audio_feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Visual encoder
        self.visual_encoder = nn.LSTM(
            input_size=visual_feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
    
    def forward(
        self,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the neural synchronization model.
        
        Args:
            audio_features: Audio features of shape (B, T_audio, D_audio).
            visual_features: Visual features of shape (B, T_visual, D_visual).
            
        Returns:
            Synchronization score of shape (B, 1).
        """
        # Encode audio features
        audio_output, _ = self.audio_encoder(audio_features)
        
        # Encode visual features
        visual_output, _ = self.visual_encoder(visual_features)
        
        # Apply attention between audio and visual features
        attended_audio, _ = self.attention(
            audio_output, visual_output, visual_output
        )
        
        # Global average pooling
        audio_pooled = torch.mean(attended_audio, dim=1)
        visual_pooled = torch.mean(visual_output, dim=1)
        
        # Concatenate features
        fused_features = torch.cat([audio_pooled, visual_pooled], dim=-1)
        
        # Predict synchronization score
        sync_score = self.fusion(fused_features)
        
        return sync_score
