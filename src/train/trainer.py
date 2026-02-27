"""
Audio-Visual Synchronization: Training utilities and loops.

This module provides training utilities and loops for audio-visual
synchronization models.
"""

from typing import Dict, Any, Optional, Tuple
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from omegaconf import DictConfig

from ..models.sync_model import AudioVisualSyncModel
from ..metrics.sync_metrics import AudioVisualSyncEvaluator
from ..utils.core import get_device, set_seed


class SynchronizationTrainer:
    """Trainer class for audio-visual synchronization models."""
    
    def __init__(
        self,
        model: AudioVisualSyncModel,
        config: DictConfig,
        device: Optional[torch.device] = None
    ):
        """Initialize the trainer.
        
        Args:
            model: The model to train.
            config: Training configuration.
            device: Device to train on.
        """
        self.model = model
        self.config = config
        self.device = device or get_device(config.device)
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize loss function
        self.criterion = self._create_criterion()
        
        # Initialize evaluator
        self.evaluator = AudioVisualSyncEvaluator(
            tolerance_frames=config.eval.tolerance_frames
        )
        
        # Initialize logging
        self.writer = None
        if hasattr(config, 'logging') and config.logging.get('use_tensorboard', False):
            log_dir = Path(config.paths.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_metric = float('inf')
        self.patience_counter = 0
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config.paths.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        optimizer_name = self.config.train.get('optimizer', 'adam').lower()
        learning_rate = self.config.train.learning_rate
        
        if optimizer_name == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=self.config.train.get('weight_decay', 1e-4)
            )
        elif optimizer_name == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=self.config.train.get('weight_decay', 1e-4)
            )
        elif optimizer_name == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=self.config.train.get('momentum', 0.9),
                weight_decay=self.config.train.get('weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _create_criterion(self) -> nn.Module:
        """Create loss function based on model type."""
        if self.model.model_name == "cross_correlation":
            # For cross-correlation model, we use correlation-based loss
            return CorrelationLoss()
        else:
            # For neural model, we use MSE loss for lag prediction
            return nn.MSELoss()
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            dataloader: Training data loader.
            
        Returns:
            Dictionary containing training metrics.
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            audio = batch['audio'].to(self.device)
            mouth_regions = batch['mouth_regions'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(audio, batch['video_frames'], mouth_regions)
            
            # Compute loss
            loss = self._compute_loss(outputs, batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.train.get('gradient_clip_val', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.train.gradient_clip_val
                )
            
            # Update parameters
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Log to tensorboard
            if self.writer and batch_idx % self.config.logging.log_every_n_steps == 0:
                global_step = self.current_epoch * len(dataloader) + batch_idx
                self.writer.add_scalar('train/loss', loss.item(), global_step)
        
        avg_loss = total_loss / num_batches
        
        return {
            'train_loss': avg_loss,
            'num_batches': num_batches
        }
    
    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch.
        
        Args:
            dataloader: Validation data loader.
            
        Returns:
            Dictionary containing validation metrics.
        """
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                audio = batch['audio'].to(self.device)
                mouth_regions = batch['mouth_regions'].to(self.device)
                
                # Forward pass
                outputs = self.model(audio, batch['video_frames'], mouth_regions)
                
                # Compute loss
                loss = self._compute_loss(outputs, batch)
                
                # Update metrics
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Evaluate model
        eval_results = self.evaluator.evaluate_model(self.model, dataloader, self.device)
        
        return {
            'val_loss': avg_loss,
            'num_batches': num_batches,
            **eval_results
        }
    
    def _compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> torch.Tensor:
        """Compute loss based on model outputs and batch.
        
        Args:
            outputs: Model outputs.
            batch: Input batch.
            
        Returns:
            Computed loss tensor.
        """
        if self.model.model_name == "cross_correlation":
            # For cross-correlation model, maximize correlation
            correlation_score = outputs['correlation_score']
            # Convert to loss (higher correlation = lower loss)
            loss = 1.0 - correlation_score.mean()
        else:
            # For neural model, predict lag offset
            predicted_lag = outputs['lag_offset']
            # Assume target lag is 0 (synchronized)
            target_lag = torch.zeros_like(predicted_lag)
            loss = self.criterion(predicted_lag, target_lag)
        
        return loss
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None
    ) -> Dict[str, Any]:
        """Train the model.
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            num_epochs: Number of epochs to train.
            
        Returns:
            Training history and best metrics.
        """
        num_epochs = num_epochs or self.config.train.num_epochs
        patience = self.config.train.patience
        
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'sync_accuracy': [],
            'mae_lag': []
        }
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model: {self.model.model_name}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            training_history['train_loss'].append(train_metrics['train_loss'])
            
            # Validate
            if val_loader is not None:
                val_metrics = self.validate_epoch(val_loader)
                training_history['val_loss'].append(val_metrics['val_loss'])
                training_history['sync_accuracy'].append(val_metrics.get('sync_accuracy', 0))
                training_history['mae_lag'].append(val_metrics.get('mae_lag', 0))
                
                # Log validation metrics
                if self.writer:
                    self.writer.add_scalar('val/loss', val_metrics['val_loss'], epoch)
                    self.writer.add_scalar('val/sync_accuracy', val_metrics.get('sync_accuracy', 0), epoch)
                    self.writer.add_scalar('val/mae_lag', val_metrics.get('mae_lag', 0), epoch)
                
                # Check for improvement
                monitor_metric = val_metrics.get(self.config.logging.monitor, val_metrics['val_loss'])
                
                if monitor_metric < self.best_metric:
                    self.best_metric = monitor_metric
                    self.patience_counter = 0
                    self.save_checkpoint('best_model.pt')
                else:
                    self.patience_counter += 1
                
                # Print epoch results
                print(f"Epoch {epoch:3d}/{num_epochs}: "
                      f"train_loss={train_metrics['train_loss']:.4f}, "
                      f"val_loss={val_metrics['val_loss']:.4f}, "
                      f"sync_acc={val_metrics.get('sync_accuracy', 0):.3f}, "
                      f"mae_lag={val_metrics.get('mae_lag', 0):.3f}")
                
                # Early stopping
                if patience > 0 and self.patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            else:
                print(f"Epoch {epoch:3d}/{num_epochs}: train_loss={train_metrics['train_loss']:.4f}")
        
        # Save final checkpoint
        self.save_checkpoint('final_model.pt')
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Close tensorboard writer
        if self.writer:
            self.writer.close()
        
        return {
            'history': training_history,
            'best_metric': self.best_metric,
            'training_time': training_time,
            'epochs_trained': epoch + 1
        }
    
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint.
        
        Args:
            filename: Checkpoint filename.
        """
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint.
        
        Args:
            filename: Checkpoint filename.
        """
        checkpoint_path = self.checkpoint_dir / filename
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint['best_metric']
        
        print(f"Checkpoint loaded from {checkpoint_path}")


class CorrelationLoss(nn.Module):
    """Loss function for correlation-based synchronization."""
    
    def __init__(self):
        """Initialize correlation loss."""
        super().__init__()
    
    def forward(self, correlation_scores: torch.Tensor) -> torch.Tensor:
        """Compute correlation loss.
        
        Args:
            correlation_scores: Correlation scores.
            
        Returns:
            Loss value.
        """
        # Maximize correlation (minimize 1 - correlation)
        return 1.0 - correlation_scores.mean()
