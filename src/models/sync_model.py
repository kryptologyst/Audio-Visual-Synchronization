"""
Audio-Visual Synchronization: Model definitions.

This module contains the main model architectures for audio-visual
synchronization tasks.
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from ..features.extractors import (
    AudioFeatureExtractor,
    VisualFeatureExtractor,
    CrossCorrelationSync,
    NeuralSyncModel
)


class CrossCorrelationModel(nn.Module):
    """Cross-correlation based audio-visual synchronization model."""
    
    def __init__(self, config: DictConfig):
        """Initialize cross-correlation model.
        
        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config
        
        # Feature extractors
        self.audio_extractor = AudioFeatureExtractor(
            sample_rate=config.audio.sample_rate,
            n_mfcc=config.audio.n_mfcc,
            n_mels=config.audio.n_mels,
            n_fft=config.audio.n_fft,
            hop_length=config.audio.hop_length,
            preemphasis=config.audio.preemphasis
        )
        
        self.visual_extractor = VisualFeatureExtractor(
            input_size=(config.video.mouth_region_size, config.video.mouth_region_size),
            feature_dim=config.model.visual_feature_dim,
            backbone="simple_cnn"
        )
        
        # Cross-correlation synchronizer
        self.sync_module = CrossCorrelationSync(normalize=True)
    
    def forward(
        self,
        audio: torch.Tensor,
        video_frames: torch.Tensor,
        mouth_regions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the cross-correlation model.
        
        Args:
            audio: Input audio tensor.
            video_frames: Input video frames.
            mouth_regions: Extracted mouth regions.
            
        Returns:
            Dictionary containing synchronization results.
        """
        batch_size = audio.shape[0] if audio.dim() > 1 else 1
        
        # Extract audio features
        audio_features = self.audio_extractor(audio)  # (T_audio, n_mfcc) or (B, T_audio, n_mfcc)
        
        # Extract visual features
        visual_features = self.visual_extractor(mouth_regions)  # (T_visual, feature_dim) or (B, T_visual, feature_dim)
        
        # Handle batch dimension
        if batch_size > 1:
            # Process each sample in the batch
            lags = []
            correlation_scores = []
            
            for i in range(batch_size):
                audio_np = audio_features[i].detach().cpu().numpy()
                visual_np = visual_features[i].detach().cpu().numpy()
                
                lag, correlation_score = self.sync_module.synchronize(audio_np, visual_np)
                lags.append(lag)
                correlation_scores.append(correlation_score)
            
            return {
                'lag': torch.tensor(lags, dtype=torch.long),
                'correlation_score': torch.tensor(correlation_scores, dtype=torch.float32),
                'audio_features': audio_features,
                'visual_features': visual_features
            }
        else:
            # Single sample
            audio_np = audio_features.detach().cpu().numpy()
            visual_np = visual_features.detach().cpu().numpy()
            
            # Compute synchronization
            lag, correlation_score = self.sync_module.synchronize(audio_np, visual_np)
            
            return {
                'lag': torch.tensor(lag, dtype=torch.long),
                'correlation_score': torch.tensor(correlation_score, dtype=torch.float32),
                'audio_features': audio_features,
                'visual_features': visual_features
            }


class NeuralSynchronizationModel(nn.Module):
    """Neural network-based audio-visual synchronization model."""
    
    def __init__(self, config: DictConfig):
        """Initialize neural synchronization model.
        
        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config
        
        # Feature extractors
        self.audio_extractor = AudioFeatureExtractor(
            sample_rate=config.audio.sample_rate,
            n_mfcc=config.audio.n_mfcc,
            n_mels=config.audio.n_mels,
            n_fft=config.audio.n_fft,
            hop_length=config.audio.hop_length,
            preemphasis=config.audio.preemphasis
        )
        
        self.visual_extractor = VisualFeatureExtractor(
            input_size=(config.video.mouth_region_size, config.video.mouth_region_size),
            feature_dim=config.model.visual_feature_dim,
            backbone="simple_cnn"
        )
        
        # Neural synchronization module
        self.sync_module = NeuralSyncModel(
            audio_feature_dim=config.audio.n_mfcc,
            visual_feature_dim=config.model.visual_feature_dim,
            hidden_dim=config.model.hidden_dim,
            dropout=config.model.dropout
        )
        
        # Lag prediction head
        self.lag_predictor = nn.Sequential(
            nn.Linear(config.model.hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(64, 1)  # Predict lag offset
        )
    
    def forward(
        self,
        audio: torch.Tensor,
        video_frames: torch.Tensor,
        mouth_regions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the neural synchronization model.
        
        Args:
            audio: Input audio tensor.
            video_frames: Input video frames.
            mouth_regions: Extracted mouth regions.
            
        Returns:
            Dictionary containing synchronization results.
        """
        batch_size = audio.shape[0] if audio.dim() > 1 else 1
        
        # Extract audio features
        audio_features = self.audio_extractor(audio)  # (T_audio, n_mfcc)
        
        # Extract visual features
        visual_features = self.visual_extractor(mouth_regions)  # (T_visual, feature_dim)
        
        # Add batch dimension if needed
        if audio_features.dim() == 2:
            audio_features = audio_features.unsqueeze(0)
        if visual_features.dim() == 2:
            visual_features = visual_features.unsqueeze(0)
        
        # Compute synchronization score
        sync_score = self.sync_module(audio_features, visual_features)
        
        # Predict lag offset
        lag_offset = self.lag_predictor(sync_score)
        
        return {
            'sync_score': sync_score,
            'lag_offset': lag_offset,
            'audio_features': audio_features,
            'visual_features': visual_features
        }


class AudioVisualSyncModel(nn.Module):
    """Main audio-visual synchronization model with multiple approaches."""
    
    def __init__(self, config: DictConfig):
        """Initialize the main synchronization model.
        
        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config
        self.model_name = config.model.name
        
        # Initialize the specified model
        if self.model_name == "cross_correlation":
            self.model = CrossCorrelationModel(config)
        elif self.model_name == "neural":
            self.model = NeuralSynchronizationModel(config)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
    
    def forward(
        self,
        audio: torch.Tensor,
        video_frames: torch.Tensor,
        mouth_regions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the main model.
        
        Args:
            audio: Input audio tensor.
            video_frames: Input video frames.
            mouth_regions: Extracted mouth regions.
            
        Returns:
            Dictionary containing model outputs.
        """
        return self.model(audio, video_frames, mouth_regions)
    
    def predict_sync(
        self,
        audio: torch.Tensor,
        video_frames: torch.Tensor,
        mouth_regions: torch.Tensor
    ) -> Tuple[int, float]:
        """Predict synchronization lag and confidence.
        
        Args:
            audio: Input audio tensor.
            video_frames: Input video frames.
            mouth_regions: Extracted mouth regions.
            
        Returns:
            Tuple of (predicted_lag, confidence_score).
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(audio, video_frames, mouth_regions)
            
            if self.model_name == "cross_correlation":
                lag = outputs['lag'].item()
                confidence = outputs['correlation_score'].item()
            else:  # neural
                lag = outputs['lag_offset'].item()
                confidence = torch.sigmoid(outputs['sync_score']).item()
            
            return int(lag), float(confidence)


def create_model(config: DictConfig) -> AudioVisualSyncModel:
    """Create a model instance from configuration.
    
    Args:
        config: Model configuration.
        
    Returns:
        Initialized model instance.
    """
    return AudioVisualSyncModel(config)
