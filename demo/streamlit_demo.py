"""
Audio-Visual Synchronization: Interactive Streamlit Demo.

This module provides an interactive demo for audio-visual synchronization
using Streamlit.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import streamlit as st
import numpy as np
import torch
import cv2
import librosa
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.sync_model import create_model
from src.utils.core import get_device, set_seed
from src.features.extractors import AudioFeatureExtractor, VisualFeatureExtractor
from src.metrics.sync_metrics import SynchronizationMetrics


# Page configuration
st.set_page_config(
    page_title="Audio-Visual Synchronization Demo",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Privacy disclaimer
PRIVACY_DISCLAIMER = """
**PRIVACY DISCLAIMER & RESEARCH USE ONLY**

This is a research demonstration tool for audio-visual synchronization. 
It is NOT intended for biometric identification or production use.

- No personal data is stored or transmitted
- All processing is done locally
- This tool is for educational and research purposes only
- Misuse for voice cloning or biometric identification is prohibited
"""


@st.cache_resource
def load_model(model_type: str = "cross_correlation"):
    """Load the synchronization model."""
    try:
        # Create a minimal config for the demo
        config = {
            'model': {'name': model_type},
            'audio': {
                'sample_rate': 16000,
                'n_mfcc': 13,
                'n_mels': 128,
                'n_fft': 2048,
                'hop_length': 512,
                'preemphasis': 0.97
            },
            'video': {
                'target_fps': 25,
                'mouth_region_size': 64
            },
            'device': 'auto'
        }
        
        # Convert to OmegaConf-like object
        class Config:
            def __init__(self, d):
                for k, v in d.items():
                    if isinstance(v, dict):
                        setattr(self, k, Config(v))
                    else:
                        setattr(self, k, v)
        
        config_obj = Config(config)
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Create model
        model = create_model(config_obj)
        model.eval()
        
        return model, config_obj
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


def extract_audio_features(audio_file, config):
    """Extract audio features from uploaded file."""
    try:
        # Load audio
        audio, sr = librosa.load(audio_file, sr=config.audio.sample_rate)
        
        # Extract MFCC features
        extractor = AudioFeatureExtractor(
            sample_rate=config.audio.sample_rate,
            n_mfcc=config.audio.n_mfcc,
            n_mels=config.audio.n_mels,
            n_fft=config.audio.n_fft,
            hop_length=config.audio.hop_length,
            preemphasis=config.audio.preemphasis
        )
        
        audio_tensor = torch.from_numpy(audio).float()
        features = extractor(audio_tensor)
        
        return audio, sr, features.numpy()
    except Exception as e:
        st.error(f"Error extracting audio features: {e}")
        return None, None, None


def extract_video_features(video_file, config):
    """Extract visual features from uploaded video."""
    try:
        # Load video frames
        cap = cv2.VideoCapture(video_file)
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        if not frames:
            raise ValueError("No frames could be loaded from video")
        
        frames = np.array(frames)
        
        # Extract mouth regions
        mouth_regions = []
        for frame in frames:
            # Simple mouth region extraction (center region for demo)
            h, w = frame.shape[:2]
            center_region = frame[h//3:2*h//3, w//3:2*w//3]
            mouth_region = cv2.resize(center_region, (config.video.mouth_region_size, config.video.mouth_region_size))
            mouth_regions.append(mouth_region)
        
        mouth_regions = np.array(mouth_regions)
        
        # Extract visual features
        extractor = VisualFeatureExtractor(
            input_size=(config.video.mouth_region_size, config.video.mouth_region_size),
            feature_dim=128,
            backbone="simple_cnn"
        )
        
        mouth_tensor = torch.from_numpy(mouth_regions).float()
        features = extractor(mouth_tensor)
        
        return frames, mouth_regions, features.numpy()
    except Exception as e:
        st.error(f"Error extracting video features: {e}")
        return None, None, None


def plot_audio_features(audio, sr, features):
    """Plot audio waveform and features."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Audio Waveform", "MFCC Features"),
        vertical_spacing=0.1
    )
    
    # Plot waveform
    time_axis = np.linspace(0, len(audio) / sr, len(audio))
    fig.add_trace(
        go.Scatter(x=time_axis, y=audio, name="Waveform", line=dict(color='blue')),
        row=1, col=1
    )
    
    # Plot MFCC features
    feature_time = np.linspace(0, len(audio) / sr, features.shape[1])
    for i in range(min(5, features.shape[0])):  # Show first 5 MFCC coefficients
        fig.add_trace(
            go.Scatter(
                x=feature_time, 
                y=features[i], 
                name=f"MFCC {i+1}",
                line=dict(width=1)
            ),
            row=2, col=1
        )
    
    fig.update_layout(height=600, showlegend=True)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="MFCC Value", row=2, col=1)
    
    return fig


def plot_video_features(frames, mouth_regions, features):
    """Plot video frames and visual features."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Original Frame", "Mouth Region", "Visual Features", "Feature Timeline"),
        specs=[[{"type": "image"}, {"type": "image"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # Show first frame
    fig.add_trace(
        go.Image(z=cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)),
        row=1, col=1
    )
    
    # Show first mouth region
    fig.add_trace(
        go.Image(z=cv2.cvtColor(mouth_regions[0], cv2.COLOR_BGR2RGB)),
        row=1, col=2
    )
    
    # Plot visual features (first few dimensions)
    feature_time = np.linspace(0, len(frames) / 25, len(frames))  # Assuming 25 FPS
    for i in range(min(3, features.shape[1])):
        fig.add_trace(
            go.Scatter(
                x=feature_time,
                y=features[:, i],
                name=f"Feature {i+1}",
                line=dict(width=2)
            ),
            row=2, col=1
        )
    
    # Plot feature magnitude over time
    feature_magnitude = np.linalg.norm(features, axis=1)
    fig.add_trace(
        go.Scatter(
            x=feature_time,
            y=feature_magnitude,
            name="Feature Magnitude",
            line=dict(color='red', width=2)
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=True)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="Feature Value", row=2, col=1)
    fig.update_yaxes(title_text="Magnitude", row=2, col=2)
    
    return fig


def compute_synchronization(audio_features, visual_features, model, config):
    """Compute synchronization between audio and visual features."""
    try:
        # Convert to tensors
        audio_tensor = torch.from_numpy(audio_features).float().unsqueeze(0)
        visual_tensor = torch.from_numpy(visual_features).float().unsqueeze(0)
        
        # Create dummy video frames and mouth regions for model input
        dummy_frames = torch.zeros(1, visual_features.shape[0], 224, 224, 3)
        dummy_mouth_regions = torch.zeros(1, visual_features.shape[0], config.video.mouth_region_size, config.video.mouth_region_size, 3)
        
        # Get device
        device = get_device(config.device)
        model = model.to(device)
        audio_tensor = audio_tensor.to(device)
        visual_tensor = visual_tensor.to(device)
        dummy_frames = dummy_frames.to(device)
        dummy_mouth_regions = dummy_mouth_regions.to(device)
        
        # Run model
        with torch.no_grad():
            outputs = model(audio_tensor, dummy_frames, dummy_mouth_regions)
        
        if 'lag' in outputs:
            # Cross-correlation model
            lag = outputs['lag'].item()
            confidence = outputs['correlation_score'].item()
        else:
            # Neural model
            lag = outputs['lag_offset'].item()
            confidence = torch.sigmoid(outputs['sync_score']).item()
        
        return int(lag), float(confidence)
    except Exception as e:
        st.error(f"Error computing synchronization: {e}")
        return 0, 0.0


def main():
    """Main demo function."""
    # Header
    st.markdown('<h1 class="main-header">🎬 Audio-Visual Synchronization Demo</h1>', unsafe_allow_html=True)
    
    # Privacy disclaimer
    st.markdown(f'<div class="warning-box">{PRIVACY_DISCLAIMER}</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["cross_correlation", "neural"],
        help="Choose the synchronization model to use"
    )
    
    # Load model
    with st.spinner("Loading model..."):
        model, config = load_model(model_type)
    
    if model is None:
        st.error("Failed to load model. Please try again.")
        return
    
    st.sidebar.success(f"✅ {model_type.replace('_', ' ').title()} model loaded")
    
    # File upload
    st.header("📁 Upload Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Audio File")
        audio_file = st.file_uploader(
            "Upload audio file",
            type=['wav', 'mp3', 'flac', 'm4a'],
            help="Upload an audio file for synchronization analysis"
        )
    
    with col2:
        st.subheader("Video File")
        video_file = st.file_uploader(
            "Upload video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file for synchronization analysis"
        )
    
    # Process files if both are uploaded
    if audio_file is not None and video_file is not None:
        st.header("🔍 Feature Extraction")
        
        # Save uploaded files temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.name.split('.')[-1]}") as tmp_audio:
            tmp_audio.write(audio_file.getvalue())
            tmp_audio_path = tmp_audio.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{video_file.name.split('.')[-1]}") as tmp_video:
            tmp_video.write(video_file.getvalue())
            tmp_video_path = tmp_video.name
        
        try:
            # Extract features
            with st.spinner("Extracting audio features..."):
                audio, sr, audio_features = extract_audio_features(tmp_audio_path, config)
            
            with st.spinner("Extracting video features..."):
                frames, mouth_regions, visual_features = extract_video_features(tmp_video_path, config)
            
            if audio_features is not None and visual_features is not None:
                st.success("✅ Features extracted successfully!")
                
                # Display feature plots
                st.header("📊 Feature Visualization")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Audio Features")
                    audio_fig = plot_audio_features(audio, sr, audio_features)
                    st.plotly_chart(audio_fig, use_container_width=True)
                
                with col2:
                    st.subheader("Video Features")
                    video_fig = plot_video_features(frames, mouth_regions, visual_features)
                    st.plotly_chart(video_fig, use_container_width=True)
                
                # Synchronization analysis
                st.header("🎯 Synchronization Analysis")
                
                with st.spinner("Computing synchronization..."):
                    lag, confidence = compute_synchronization(audio_features, visual_features, model, config)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Synchronization Lag",
                        f"{lag} frames",
                        help="Number of frames the audio is offset from video"
                    )
                
                with col2:
                    st.metric(
                        "Confidence Score",
                        f"{confidence:.3f}",
                        help="Confidence in the synchronization prediction"
                    )
                
                with col3:
                    lag_ms = lag * (1000 / config.video.target_fps)  # Convert to milliseconds
                    st.metric(
                        "Lag (milliseconds)",
                        f"{lag_ms:.1f} ms",
                        help="Time offset in milliseconds"
                    )
                
                # Interpretation
                st.subheader("📝 Interpretation")
                
                if abs(lag) <= 2:
                    st.success("✅ **Well Synchronized**: Audio and video are properly aligned")
                elif abs(lag) <= 5:
                    st.warning("⚠️ **Slightly Out of Sync**: Minor synchronization issues detected")
                else:
                    st.error("❌ **Poorly Synchronized**: Significant synchronization problems detected")
                
                # Recommendations
                st.subheader("💡 Recommendations")
                
                if lag > 0:
                    st.info(f"**Audio is ahead by {lag} frames**. Consider delaying the audio by {lag_ms:.1f}ms.")
                elif lag < 0:
                    st.info(f"**Video is ahead by {abs(lag)} frames**. Consider delaying the video by {abs(lag_ms):.1f}ms.")
                else:
                    st.info("**Perfect synchronization!** No adjustments needed.")
                
                # Technical details
                with st.expander("🔧 Technical Details"):
                    st.write(f"**Audio Sample Rate**: {sr} Hz")
                    st.write(f"**Video Frame Rate**: {config.video.target_fps} FPS")
                    st.write(f"**Audio Duration**: {len(audio) / sr:.2f} seconds")
                    st.write(f"**Video Duration**: {len(frames) / config.video.target_fps:.2f} seconds")
                    st.write(f"**Audio Features Shape**: {audio_features.shape}")
                    st.write(f"**Visual Features Shape**: {visual_features.shape}")
                    st.write(f"**Model Type**: {model_type.replace('_', ' ').title()}")
        
        finally:
            # Clean up temporary files
            os.unlink(tmp_audio_path)
            os.unlink(tmp_video_path)
    
    elif audio_file is not None or video_file is not None:
        st.warning("⚠️ Please upload both audio and video files to perform synchronization analysis.")
    
    # Demo instructions
    st.header("📖 How to Use")
    st.markdown("""
    1. **Upload Files**: Upload both an audio file (.wav, .mp3, .flac, .m4a) and a video file (.mp4, .avi, .mov, .mkv)
    2. **Feature Extraction**: The system will automatically extract audio (MFCC) and visual (mouth region) features
    3. **Synchronization Analysis**: The model will compute the optimal synchronization lag between audio and video
    4. **Results**: View the synchronization lag, confidence score, and recommendations for alignment
    
    **Note**: This demo works best with speech content where mouth movements correlate with audio.
    """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Audio-Visual Synchronization Demo** | "
        "Research Use Only | "
        "Privacy-Preserving | "
        "No Data Storage"
    )


if __name__ == "__main__":
    main()
