#!/usr/bin/env python3
"""
Audio-Visual Synchronization: Main training script.

This script provides the main entry point for training audio-visual
synchronization models.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.sync_model import create_model
from src.train.trainer import SynchronizationTrainer
from src.utils.core import AudioVisualSyncDataset, set_seed, get_device, create_synthetic_dataset
from src.metrics.sync_metrics import create_leaderboard


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Audio-Visual Synchronization Model")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Path to processed data directory"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices=["cross_correlation", "neural"],
        help="Model type to train"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use for training"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed"
    )
    
    parser.add_argument(
        "--create_synthetic",
        action="store_true",
        help="Create synthetic dataset for training"
    )
    
    parser.add_argument(
        "--synthetic_samples",
        type=int,
        default=100,
        help="Number of synthetic samples to create"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume from"
    )
    
    return parser.parse_args()


def create_data_loaders(config, data_dir):
    """Create data loaders for training and validation.
    
    Args:
        config: Configuration object.
        data_dir: Path to data directory.
        
    Returns:
        Tuple of (train_loader, val_loader).
    """
    # Create datasets
    train_dataset = AudioVisualSyncDataset(
        data_dir=data_dir,
        config=config,
        split="train"
    )
    
    val_dataset = AudioVisualSyncDataset(
        data_dir=data_dir,
        config=config,
        split="val"
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues on macOS
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues on macOS
        pin_memory=True
    )
    
    return train_loader, val_loader


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Override config with command line arguments
    if args.model:
        config.model.name = args.model
    if args.epochs:
        config.train.num_epochs = args.epochs
    if args.batch_size:
        config.train.batch_size = args.batch_size
    if args.learning_rate:
        config.train.learning_rate = args.learning_rate
    if args.device:
        config.device = args.device
    if args.seed:
        config.seed = args.seed
    
    # Set random seed
    set_seed(config.seed)
    
    # Get device
    device = get_device(config.device)
    print(f"Using device: {device}")
    
    # Create synthetic dataset if requested
    if args.create_synthetic:
        print("Creating synthetic dataset...")
        create_synthetic_dataset(
            output_dir=args.data_dir,
            num_samples=args.synthetic_samples,
            duration=3.0,
            sample_rate=config.audio.sample_rate,
            fps=config.video.target_fps
        )
        print(f"Synthetic dataset created with {args.synthetic_samples} samples")
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Data directory {args.data_dir} does not exist.")
        print("Please provide a valid data directory or use --create_synthetic to generate synthetic data.")
        return
    
    # Create data loaders
    print("Creating data loaders...")
    try:
        train_loader, val_loader = create_data_loaders(config, args.data_dir)
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        return
    
    # Create model
    print(f"Creating {config.model.name} model...")
    model = create_model(config)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = SynchronizationTrainer(model, config, device)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train model
    print("Starting training...")
    results = trainer.train(train_loader, val_loader)
    
    # Print training summary
    print("\nTraining Summary:")
    print("=" * 50)
    print(f"Epochs trained: {results['epochs_trained']}")
    print(f"Training time: {results['training_time']:.2f} seconds")
    print(f"Best metric: {results['best_metric']:.4f}")
    
    # Print final validation metrics
    print("\nFinal Validation Metrics:")
    print(trainer.evaluator.get_summary())
    
    # Create leaderboard
    leaderboard_results = {
        f"{config.model.name}_model": trainer.evaluator.sync_metrics.compute_metrics()
    }
    leaderboard = create_leaderboard(leaderboard_results)
    print(f"\n{leaderboard}")
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
