"""
OpenWorld-Multimodal Physics Metrics
Physics-informed evaluation metrics for world modeling.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple

from ..utils.logging import get_logger

logger = get_logger(__name__)


class PhysicsMetrics:
    """
    Physics-informed metrics for evaluating world model consistency.
    
    Evaluates:
    - Motion consistency
    - Momentum conservation
    - Velocity smoothness
    - Acceleration patterns
    - Spatial coherence
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        
        logger.info("PhysicsMetrics initialized")
    
    def compute_physics_consistency(
        self,
        predicted_video: torch.Tensor,
        latent_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Compute physics consistency metrics.
        
        Args:
            predicted_video: Predicted video sequence (B, T, C, H, W)
            latent_features: Optional latent features for analysis
            
        Returns:
            Dictionary of physics metrics
        """
        metrics = {}
        
        # Motion consistency
        metrics.update(self.compute_motion_consistency(predicted_video))
        
        # Velocity analysis
        metrics.update(self.compute_velocity_metrics(predicted_video))
        
        # Acceleration analysis
        metrics.update(self.compute_acceleration_metrics(predicted_video))
        
        # Spatial coherence
        metrics.update(self.compute_spatial_coherence(predicted_video))
        
        # Energy conservation (simplified)
        metrics.update(self.compute_energy_metrics(predicted_video))
        
        return metrics
    
    def compute_motion_consistency(self, video: torch.Tensor) -> Dict[str, float]:
        """Compute motion consistency metrics."""
        B, T, C, H, W = video.shape
        
        # Compute optical flow approximation
        flow = self._compute_optical_flow(video)
        
        # Motion smoothness
        flow_diff = flow[:, 1:] - flow[:, :-1]
        motion_smoothness = torch.norm(flow_diff, dim=(-3, -2, -1)).mean().item()
        
        # Motion magnitude consistency
        flow_magnitude = torch.norm(flow, dim=-3)
        magnitude_variance = torch.var(flow_magnitude, dim=1).mean().item()
        
        return {
            'motion_smoothness': motion_smoothness,
            'motion_magnitude_variance': magnitude_variance,
        }
    
    def compute_velocity_metrics(self, video: torch.Tensor) -> Dict[str, float]:
        """Compute velocity-related metrics."""
        B, T, C, H, W = video.shape
        
        # Frame differences as velocity proxy
        velocity = video[:, 1:] - video[:, :-1]
        
        # Velocity magnitude
        velocity_magnitude = torch.norm(velocity, dim=(2, 3, 4))
        
        # Velocity smoothness
        velocity_smoothness = torch.std(velocity_magnitude, dim=1).mean().item()
        
        # Average velocity
        avg_velocity = velocity_magnitude.mean().item()
        
        # Velocity direction consistency
        velocity_norm = F.normalize(velocity.flatten(2), dim=2, eps=1e-8)
        direction_consistency = torch.sum(
            velocity_norm[:, :-1] * velocity_norm[:, 1:], dim=2
        ).mean().item()
        
        return {
            'velocity_smoothness': velocity_smoothness,
            'average_velocity': avg_velocity,
            'velocity_direction_consistency': direction_consistency,
        }
    
    def compute_acceleration_metrics(self, video: torch.Tensor) -> Dict[str, float]:
        """Compute acceleration-related metrics."""
        B, T, C, H, W = video.shape
        
        if T < 3:
            return {'acceleration_smoothness': 0.0}
        
        # Second-order differences as acceleration proxy
        velocity = video[:, 1:] - video[:, :-1]
        acceleration = velocity[:, 1:] - velocity[:, :-1]
        
        # Acceleration magnitude
        acceleration_magnitude = torch.norm(acceleration, dim=(2, 3, 4))
        
        # Acceleration smoothness
        acceleration_smoothness = torch.std(acceleration_magnitude, dim=1).mean().item()
        
        # Average acceleration
        avg_acceleration = acceleration_magnitude.mean().item()
        
        return {
            'acceleration_smoothness': acceleration_smoothness,
            'average_acceleration': avg_acceleration,
        }
    
    def compute_spatial_coherence(self, video: torch.Tensor) -> Dict[str, float]:
        """Compute spatial coherence metrics."""
        B, T, C, H, W = video.shape
        
        # Spatial gradients
        grad_x = video[:, :, :, :, 1:] - video[:, :, :, :, :-1]
        grad_y = video[:, :, :, 1:, :] - video[:, :, :, :-1, :]
        
        # Spatial smoothness
        spatial_smoothness_x = torch.abs(grad_x).mean().item()
        spatial_smoothness_y = torch.abs(grad_y).mean().item()
        
        # Edge consistency over time
        edge_magnitude_x = torch.norm(grad_x, dim=2)
        edge_magnitude_y = torch.norm(grad_y, dim=2)
        
        edge_consistency_x = self._compute_temporal_consistency(edge_magnitude_x)
        edge_consistency_y = self._compute_temporal_consistency(edge_magnitude_y)
        
        return {
            'spatial_smoothness_x': spatial_smoothness_x,
            'spatial_smoothness_y': spatial_smoothness_y,
            'edge_consistency_x': edge_consistency_x,
            'edge_consistency_y': edge_consistency_y,
        }
    
    def compute_energy_metrics(self, video: torch.Tensor) -> Dict[str, float]:
        """Compute energy-related metrics (simplified)."""
        B, T, C, H, W = video.shape
        
        # Kinetic energy proxy (based on frame differences)
        velocity = video[:, 1:] - video[:, :-1]
        kinetic_energy = torch.sum(velocity ** 2, dim=(2, 3, 4))
        
        # Potential energy proxy (based on pixel intensities)
        potential_energy = torch.sum(video ** 2, dim=(2, 3, 4))
        
        # Energy conservation (should remain relatively constant)
        total_energy = kinetic_energy + potential_energy[:, 1:]
        energy_variance = torch.var(total_energy, dim=1).mean().item()
        
        # Energy smoothness
        energy_smoothness = torch.std(total_energy, dim=1).mean().item()
        
        return {
            'energy_variance': energy_variance,
            'energy_smoothness': energy_smoothness,
            'avg_kinetic_energy': kinetic_energy.mean().item(),
            'avg_potential_energy': potential_energy.mean().item(),
        }
    
    def _compute_optical_flow(self, video: torch.Tensor) -> torch.Tensor:
        """Compute simplified optical flow between consecutive frames."""
        B, T, C, H, W = video.shape
        
        if T < 2:
            return torch.zeros(B, 0, 2, H, W, device=video.device)
        
        # Convert to grayscale if needed
        if C == 3:
            gray_video = 0.299 * video[:, :, 0] + 0.587 * video[:, :, 1] + 0.114 * video[:, :, 2]
            gray_video = gray_video.unsqueeze(2)
        else:
            gray_video = video
        
        # Compute gradients
        grad_x = gray_video[:, :, :, :, 1:] - gray_video[:, :, :, :, :-1]
        grad_y = gray_video[:, :, :, 1:, :] - gray_video[:, :, :, :-1, :]
        grad_t = gray_video[:, 1:] - gray_video[:, :-1]
        
        # Pad gradients to match dimensions
        grad_x = F.pad(grad_x, (0, 1))
        grad_y = F.pad(grad_y, (0, 0, 0, 1))
        grad_t = grad_t[:, :, :, :H-1, :W-1] if H > 1 and W > 1 else grad_t
        
        # Simplified optical flow using gradient method
        # This is a very basic approximation
        flow_x = -grad_t / (grad_x[:, 1:, :, :H-1, :W-1] + 1e-8)
        flow_y = -grad_t / (grad_y[:, 1:, :, :H-1, :W-1] + 1e-8)
        
        # Pad to original size
        flow_x = F.pad(flow_x, (0, 1, 0, 1))
        flow_y = F.pad(flow_y, (0, 1, 0, 1))
        
        # Stack flow components
        flow = torch.stack([flow_x, flow_y], dim=2)
        
        return flow
    
    def _compute_temporal_consistency(self, tensor: torch.Tensor) -> float:
        """Compute temporal consistency of a tensor sequence."""
        if tensor.size(1) < 2:
            return 0.0
        
        # Compute frame-to-frame correlation
        correlations = []
        for t in range(tensor.size(1) - 1):
            corr = F.cosine_similarity(
                tensor[:, t].flatten(1),
                tensor[:, t + 1].flatten(1),
                dim=1
            )
            correlations.append(corr)
        
        correlations = torch.stack(correlations, dim=1)
        return correlations.mean().item()
    
    def compute_momentum_conservation(
        self,
        predicted_objects: List[torch.Tensor],
        masses: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Compute momentum conservation for tracked objects.
        
        Args:
            predicted_objects: List of object trajectories
            masses: Optional object masses
            
        Returns:
            Momentum conservation score
        """
        if not predicted_objects:
            return 0.0
        
        if masses is None:
            masses = torch.ones(len(predicted_objects))
        
        # Compute velocities
        total_momentum = torch.zeros(2)
        
        for obj_traj, mass in zip(predicted_objects, masses):
            if obj_traj.size(0) < 2:
                continue
            
            velocity = obj_traj[1:] - obj_traj[:-1]
            momentum = mass * velocity
            total_momentum += momentum.sum(dim=0)
        
        # Momentum should be conserved (remain constant)
        momentum_variance = torch.var(total_momentum).item()
        
        return 1.0 / (1.0 + momentum_variance)  # Higher score for better conservation


def compute_physics_score(
    predicted_video: torch.Tensor,
    device: torch.device,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute overall physics consistency score.
    
    Args:
        predicted_video: Predicted video sequence
        device: Device for computation
        weights: Optional weights for different metrics
        
    Returns:
        Overall physics score (0-1, higher is better)
    """
    if weights is None:
        weights = {
            'motion_smoothness': 0.2,
            'velocity_smoothness': 0.2,
            'acceleration_smoothness': 0.2,
            'spatial_smoothness': 0.2,
            'energy_conservation': 0.2,
        }
    
    physics_metrics = PhysicsMetrics(device)
    metrics = physics_metrics.compute_physics_consistency(predicted_video)
    
    # Normalize metrics to 0-1 scale (higher is better)
    normalized_scores = {}
    
    # Motion smoothness (lower is better, so invert)
    normalized_scores['motion_smoothness'] = 1.0 / (1.0 + metrics.get('motion_smoothness', 1.0))
    
    # Velocity smoothness (lower is better, so invert)
    normalized_scores['velocity_smoothness'] = 1.0 / (1.0 + metrics.get('velocity_smoothness', 1.0))
    
    # Acceleration smoothness (lower is better, so invert)
    normalized_scores['acceleration_smoothness'] = 1.0 / (1.0 + metrics.get('acceleration_smoothness', 1.0))
    
    # Spatial smoothness (lower is better, so invert)
    spatial_score = (
        1.0 / (1.0 + metrics.get('spatial_smoothness_x', 1.0)) +
        1.0 / (1.0 + metrics.get('spatial_smoothness_y', 1.0))
    ) / 2.0
    normalized_scores['spatial_smoothness'] = spatial_score
    
    # Energy conservation (lower variance is better, so invert)
    normalized_scores['energy_conservation'] = 1.0 / (1.0 + metrics.get('energy_variance', 1.0))
    
    # Compute weighted average
    total_score = sum(
        weights.get(key, 0.0) * score
        for key, score in normalized_scores.items()
    )
    
    return total_score 