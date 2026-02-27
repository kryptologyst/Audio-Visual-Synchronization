"""
Audio-Visual Synchronization: Evaluation metrics and utilities.

This module provides evaluation metrics specific to audio-visual
synchronization tasks.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error


class SynchronizationMetrics:
    """Metrics for audio-visual synchronization evaluation."""
    
    def __init__(self, tolerance_frames: int = 2):
        """Initialize synchronization metrics.
        
        Args:
            tolerance_frames: Tolerance in frames for considering predictions correct.
        """
        self.tolerance_frames = tolerance_frames
        self.reset()
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.confidences = []
        self.lag_errors = []
    
    def update(
        self,
        predicted_lag: int,
        target_lag: int,
        confidence: float = 1.0
    ) -> None:
        """Update metrics with a single prediction.
        
        Args:
            predicted_lag: Predicted synchronization lag.
            target_lag: Ground truth synchronization lag.
            confidence: Prediction confidence score.
        """
        self.predictions.append(predicted_lag)
        self.targets.append(target_lag)
        self.confidences.append(confidence)
        
        # Compute lag error
        lag_error = abs(predicted_lag - target_lag)
        self.lag_errors.append(lag_error)
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute all synchronization metrics.
        
        Returns:
            Dictionary containing computed metrics.
        """
        if not self.predictions:
            return {}
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        confidences = np.array(self.confidences)
        lag_errors = np.array(self.lag_errors)
        
        # Synchronization accuracy (within tolerance)
        correct_mask = lag_errors <= self.tolerance_frames
        sync_accuracy = np.mean(correct_mask)
        
        # Mean absolute lag error
        mae_lag = np.mean(lag_errors)
        
        # Root mean square lag error
        rmse_lag = np.sqrt(np.mean(lag_errors ** 2))
        
        # Perfect synchronization accuracy (exact match)
        perfect_accuracy = np.mean(lag_errors == 0)
        
        # Confidence-weighted accuracy
        if np.sum(confidences) > 0:
            weighted_accuracy = np.sum(correct_mask * confidences) / np.sum(confidences)
        else:
            weighted_accuracy = sync_accuracy
        
        # Lag error statistics
        lag_error_std = np.std(lag_errors)
        lag_error_median = np.median(lag_errors)
        
        # Percentiles
        lag_error_75th = np.percentile(lag_errors, 75)
        lag_error_90th = np.percentile(lag_errors, 90)
        lag_error_95th = np.percentile(lag_errors, 95)
        
        return {
            'sync_accuracy': float(sync_accuracy),
            'perfect_accuracy': float(perfect_accuracy),
            'weighted_accuracy': float(weighted_accuracy),
            'mae_lag': float(mae_lag),
            'rmse_lag': float(rmse_lag),
            'lag_error_std': float(lag_error_std),
            'lag_error_median': float(lag_error_median),
            'lag_error_75th': float(lag_error_75th),
            'lag_error_90th': float(lag_error_90th),
            'lag_error_95th': float(lag_error_95th),
            'num_samples': len(self.predictions)
        }


class CorrelationMetrics:
    """Metrics for correlation-based synchronization."""
    
    def __init__(self):
        """Initialize correlation metrics."""
        self.reset()
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.correlation_scores = []
        self.max_correlations = []
    
    def update(self, correlation_score: float, max_correlation: float) -> None:
        """Update correlation metrics.
        
        Args:
            correlation_score: Computed correlation score.
            max_correlation: Maximum possible correlation score.
        """
        self.correlation_scores.append(correlation_score)
        self.max_correlations.append(max_correlation)
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute correlation-based metrics.
        
        Returns:
            Dictionary containing computed metrics.
        """
        if not self.correlation_scores:
            return {}
        
        correlation_scores = np.array(self.correlation_scores)
        max_correlations = np.array(self.max_correlations)
        
        # Normalized correlation scores
        normalized_scores = correlation_scores / (max_correlations + 1e-8)
        
        return {
            'mean_correlation': float(np.mean(correlation_scores)),
            'std_correlation': float(np.std(correlation_scores)),
            'mean_normalized_correlation': float(np.mean(normalized_scores)),
            'max_correlation': float(np.max(correlation_scores)),
            'min_correlation': float(np.min(correlation_scores)),
            'correlation_median': float(np.median(correlation_scores))
        }


class AudioVisualSyncEvaluator:
    """Comprehensive evaluator for audio-visual synchronization models."""
    
    def __init__(self, tolerance_frames: int = 2):
        """Initialize the evaluator.
        
        Args:
            tolerance_frames: Tolerance in frames for synchronization accuracy.
        """
        self.tolerance_frames = tolerance_frames
        self.sync_metrics = SynchronizationMetrics(tolerance_frames)
        self.correlation_metrics = CorrelationMetrics()
    
    def evaluate_batch(
        self,
        predictions: List[Tuple[int, float]],
        targets: List[int],
        correlation_scores: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """Evaluate a batch of predictions.
        
        Args:
            predictions: List of (predicted_lag, confidence) tuples.
            targets: List of ground truth lags.
            correlation_scores: Optional list of correlation scores.
            
        Returns:
            Dictionary containing evaluation metrics.
        """
        # Update synchronization metrics
        for (pred_lag, confidence), target_lag in zip(predictions, targets):
            self.sync_metrics.update(pred_lag, target_lag, confidence)
        
        # Update correlation metrics if provided
        if correlation_scores is not None:
            for score in correlation_scores:
                self.correlation_metrics.update(score, 1.0)  # Assuming max correlation is 1.0
        
        # Compute metrics
        sync_results = self.sync_metrics.compute_metrics()
        correlation_results = self.correlation_metrics.compute_metrics()
        
        # Combine results
        results = {**sync_results, **correlation_results}
        
        return results
    
    def evaluate_model(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Dict[str, float]:
        """Evaluate a model on a dataset.
        
        Args:
            model: The model to evaluate.
            dataloader: DataLoader containing evaluation data.
            device: Device to run evaluation on.
            
        Returns:
            Dictionary containing evaluation metrics.
        """
        model.eval()
        
        all_predictions = []
        all_targets = []
        all_correlations = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                audio = batch['audio'].to(device)
                mouth_regions = batch['mouth_regions'].to(device)
                
                # Get model predictions
                outputs = model(audio, batch['video_frames'], mouth_regions)
                
                # Extract predictions based on model type
                if 'lag' in outputs:
                    # Cross-correlation model
                    predicted_lags = outputs['lag'].cpu().numpy()
                    confidences = outputs['correlation_score'].cpu().numpy()
                    correlations = confidences
                else:
                    # Neural model
                    predicted_lags = outputs['lag_offset'].cpu().numpy()
                    confidences = torch.sigmoid(outputs['sync_score']).cpu().numpy()
                    correlations = confidences
                
                # For now, assume target lag is 0 (synchronized)
                target_lags = np.zeros_like(predicted_lags)
                
                # Store results
                for pred_lag, conf, corr in zip(predicted_lags, confidences, correlations):
                    all_predictions.append((int(pred_lag), float(conf)))
                    all_correlations.append(float(corr))
                
                all_targets.extend(target_lags.tolist())
        
        # Evaluate batch
        results = self.evaluate_batch(all_predictions, all_targets, all_correlations)
        
        return results
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.sync_metrics.reset()
        self.correlation_metrics.reset()
    
    def get_summary(self) -> str:
        """Get a summary of current metrics.
        
        Returns:
            String summary of metrics.
        """
        sync_results = self.sync_metrics.compute_metrics()
        correlation_results = self.correlation_metrics.compute_metrics()
        
        summary = "Audio-Visual Synchronization Evaluation Results:\n"
        summary += "=" * 50 + "\n"
        
        if sync_results:
            summary += f"Synchronization Accuracy (±{self.tolerance_frames} frames): {sync_results['sync_accuracy']:.3f}\n"
            summary += f"Perfect Accuracy (exact match): {sync_results['perfect_accuracy']:.3f}\n"
            summary += f"Mean Absolute Lag Error: {sync_results['mae_lag']:.3f} frames\n"
            summary += f"RMSE Lag Error: {sync_results['rmse_lag']:.3f} frames\n"
            summary += f"Lag Error Std: {sync_results['lag_error_std']:.3f} frames\n"
        
        if correlation_results:
            summary += f"Mean Correlation Score: {correlation_results['mean_correlation']:.3f}\n"
            summary += f"Correlation Std: {correlation_results['std_correlation']:.3f}\n"
        
        return summary


def create_leaderboard(results: Dict[str, Dict[str, float]]) -> str:
    """Create a leaderboard from evaluation results.
    
    Args:
        results: Dictionary mapping model names to their evaluation results.
        
    Returns:
        Formatted leaderboard string.
    """
    if not results:
        return "No results available for leaderboard."
    
    # Define metrics to display
    primary_metrics = ['sync_accuracy', 'mae_lag', 'rmse_lag', 'mean_correlation']
    
    leaderboard = "Audio-Visual Synchronization Leaderboard\n"
    leaderboard += "=" * 60 + "\n"
    
    # Header
    header = f"{'Model':<20} {'Sync Acc':<10} {'MAE Lag':<10} {'RMSE Lag':<10} {'Correlation':<12}"
    leaderboard += header + "\n"
    leaderboard += "-" * 60 + "\n"
    
    # Sort models by sync accuracy (descending)
    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1].get('sync_accuracy', 0),
        reverse=True
    )
    
    # Add rows
    for model_name, metrics in sorted_models:
        sync_acc = metrics.get('sync_accuracy', 0)
        mae_lag = metrics.get('mae_lag', 0)
        rmse_lag = metrics.get('rmse_lag', 0)
        correlation = metrics.get('mean_correlation', 0)
        
        row = f"{model_name:<20} {sync_acc:<10.3f} {mae_lag:<10.3f} {rmse_lag:<10.3f} {correlation:<12.3f}"
        leaderboard += row + "\n"
    
    return leaderboard
