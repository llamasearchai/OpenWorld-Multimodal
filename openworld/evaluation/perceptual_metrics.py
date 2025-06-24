"""
OpenWorld-Multimodal Perceptual Metrics
Advanced metrics for evaluating video and audio quality.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import torchvision.transforms as transforms
from torchvision.models import vgg19
import math

from ..utils.logging import get_logger

logger = get_logger(__name__)


class PerceptualMetrics:
    """
    Comprehensive perceptual metrics for multimodal evaluation.
    
    Supports:
    - Video quality metrics (PSNR, SSIM, LPIPS)
    - Audio quality metrics (spectral distances, perceptual loss)
    - Cross-modal alignment metrics
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        
        # Initialize VGG for perceptual loss
        self.vgg = vgg19(pretrained=True).features.to(device)
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad_(False)
            
        # VGG normalization
        self.vgg_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.vgg_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        
        logger.info("PerceptualMetrics initialized")
    
    def compute_video_metrics(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute comprehensive video quality metrics.
        
        Args:
            pred_video: Predicted video (B, T, C, H, W)
            target_video: Target video (B, T, C, H, W)
            
        Returns:
            Dictionary of video metrics
        """
        metrics = {}
        
        # PSNR
        metrics['psnr'] = self.compute_psnr(pred_video, target_video)
        
        # SSIM
        metrics['ssim'] = self.compute_ssim(pred_video, target_video)
        
        # LPIPS (Perceptual distance)
        metrics['lpips'] = self.compute_lpips(pred_video, target_video)
        
        # Temporal consistency
        metrics['temporal_consistency'] = self.compute_temporal_consistency(pred_video, target_video)
        
        # Optical flow consistency
        metrics['flow_consistency'] = self.compute_flow_consistency(pred_video, target_video)
        
        return metrics
    
    def compute_audio_metrics(
        self,
        pred_audio: torch.Tensor,
        target_audio: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute comprehensive audio quality metrics.
        
        Args:
            pred_audio: Predicted audio spectrograms (B, T, F)
            target_audio: Target audio spectrograms (B, T, F)
            
        Returns:
            Dictionary of audio metrics
        """
        metrics = {}
        
        # Spectral distance
        metrics['spectral_distance'] = self.compute_spectral_distance(pred_audio, target_audio)
        
        # Spectral convergence
        metrics['spectral_convergence'] = self.compute_spectral_convergence(pred_audio, target_audio)
        
        # Log-spectral distance
        metrics['log_spectral_distance'] = self.compute_log_spectral_distance(pred_audio, target_audio)
        
        # Mel-scale perceptual distance
        metrics['mel_distance'] = self.compute_mel_distance(pred_audio, target_audio)
        
        # Temporal audio consistency
        metrics['audio_temporal_consistency'] = self.compute_audio_temporal_consistency(pred_audio, target_audio)
        
        return metrics
    
    def compute_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute Peak Signal-to-Noise Ratio."""
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()
    
    def compute_ssim(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        window_size: int = 11,
        sigma: float = 1.5,
    ) -> float:
        """Compute Structural Similarity Index."""
        # Create Gaussian window
        coords = torch.arange(window_size, dtype=torch.float32, device=pred.device)
        coords -= window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        window = g.outer(g).unsqueeze(0).unsqueeze(0)
        
        # Flatten batch and time dimensions
        pred_flat = pred.view(-1, *pred.shape[-3:])
        target_flat = target.view(-1, *target.shape[-3:])
        
        # Convert to grayscale if needed
        if pred_flat.size(1) == 3:
            pred_gray = 0.299 * pred_flat[:, 0] + 0.587 * pred_flat[:, 1] + 0.114 * pred_flat[:, 2]
            target_gray = 0.299 * target_flat[:, 0] + 0.587 * target_flat[:, 1] + 0.114 * target_flat[:, 2]
            pred_gray = pred_gray.unsqueeze(1)
            target_gray = target_gray.unsqueeze(1)
        else:
            pred_gray = pred_flat
            target_gray = target_flat
        
        # Compute SSIM
        mu1 = F.conv2d(pred_gray, window, padding=window_size // 2)
        mu2 = F.conv2d(target_gray, window, padding=window_size // 2)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(pred_gray * pred_gray, window, padding=window_size // 2) - mu1_sq
        sigma2_sq = F.conv2d(target_gray * target_gray, window, padding=window_size // 2) - mu2_sq
        sigma12 = F.conv2d(pred_gray * target_gray, window, padding=window_size // 2) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean().item()
    
    def compute_lpips(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute Learned Perceptual Image Patch Similarity using VGG."""
        # Flatten batch and time dimensions
        pred_flat = pred.view(-1, *pred.shape[-3:])
        target_flat = target.view(-1, *target.shape[-3:])
        
        # Normalize for VGG
        pred_norm = (pred_flat - self.vgg_mean) / self.vgg_std
        target_norm = (target_flat - self.vgg_mean) / self.vgg_std
        
        # Extract features
        pred_features = []
        target_features = []
        
        x_pred = pred_norm
        x_target = target_norm
        
        for layer in self.vgg:
            x_pred = layer(x_pred)
            x_target = layer(x_target)
            
            if isinstance(layer, nn.ReLU):
                pred_features.append(x_pred)
                target_features.append(x_target)
        
        # Compute perceptual distance
        lpips_distance = 0
        for pred_feat, target_feat in zip(pred_features, target_features):
            lpips_distance += F.mse_loss(pred_feat, target_feat)
        
        return lpips_distance.item()
    
    def compute_temporal_consistency(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute temporal consistency between consecutive frames."""
        # Compute frame differences
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        
        # Compute consistency as similarity of differences
        consistency = F.cosine_similarity(
            pred_diff.flatten(2), target_diff.flatten(2), dim=2
        )
        
        return consistency.mean().item()
    
    def compute_flow_consistency(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute optical flow consistency (simplified version)."""
        # Simplified flow consistency using frame differences
        pred_flow = pred[:, 1:] - pred[:, :-1]
        target_flow = target[:, 1:] - target[:, :-1]
        
        # Compute flow magnitude consistency
        pred_mag = torch.norm(pred_flow, dim=2, keepdim=True)
        target_mag = torch.norm(target_flow, dim=2, keepdim=True)
        
        flow_consistency = F.l1_loss(pred_mag, target_mag)
        
        return flow_consistency.item()
    
    def compute_spectral_distance(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute spectral distance between audio spectrograms."""
        return F.l1_loss(pred, target).item()
    
    def compute_spectral_convergence(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute spectral convergence metric."""
        numerator = torch.norm(pred - target, p='fro')
        denominator = torch.norm(target, p='fro')
        
        return (numerator / (denominator + 1e-8)).item()
    
    def compute_log_spectral_distance(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute log-spectral distance."""
        log_pred = torch.log(pred + 1e-8)
        log_target = torch.log(target + 1e-8)
        
        return F.mse_loss(log_pred, log_target).item()
    
    def compute_mel_distance(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute mel-scale perceptual distance."""
        # Convert to mel scale (simplified)
        mel_pred = self._to_mel_scale(pred)
        mel_target = self._to_mel_scale(target)
        
        return F.l1_loss(mel_pred, mel_target).item()
    
    def compute_audio_temporal_consistency(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute temporal consistency for audio spectrograms."""
        # Compute temporal differences
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        
        # Compute consistency
        consistency = F.cosine_similarity(
            pred_diff.flatten(1), target_diff.flatten(1), dim=1
        )
        
        return consistency.mean().item()
    
    def _to_mel_scale(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Convert spectrogram to mel scale (simplified)."""
        # Simplified mel scale conversion
        # In practice, you'd use a proper mel filter bank
        mel_filters = torch.linspace(0, 1, spectrogram.size(-1), device=spectrogram.device)
        mel_filters = mel_filters.unsqueeze(0).unsqueeze(0)
        
        return spectrogram * mel_filters
    
    def compute_fvd(self, pred_video: torch.Tensor, target_video: torch.Tensor) -> float:
        """
        Compute Fréchet Video Distance (simplified version).
        
        Args:
            pred_video: Predicted video (B, T, C, H, W)
            target_video: Target video (B, T, C, H, W)
            
        Returns:
            FVD score
        """
        # Extract features using VGG
        pred_features = self._extract_video_features(pred_video)
        target_features = self._extract_video_features(target_video)
        
        # Compute Fréchet distance
        mu1, sigma1 = pred_features.mean(0), torch.cov(pred_features.T)
        mu2, sigma2 = target_features.mean(0), torch.cov(target_features.T)
        
        diff = mu1 - mu2
        
        # Compute matrix square root (simplified)
        sqrt_sigma = torch.sqrt(torch.diag(sigma1 * sigma2))
        
        fvd = torch.sum(diff ** 2) + torch.trace(sigma1 + sigma2 - 2 * sqrt_sigma)
        
        return fvd.item()
    
    def _extract_video_features(self, video: torch.Tensor) -> torch.Tensor:
        """Extract features from video for FVD computation."""
        B, T, C, H, W = video.shape
        
        # Reshape to (B*T, C, H, W)
        video_flat = video.view(B * T, C, H, W)
        
        # Normalize for VGG
        video_norm = (video_flat - self.vgg_mean) / self.vgg_std
        
        # Extract features
        with torch.no_grad():
            features = self.vgg(video_norm)
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(B * T, -1)
        
        return features
    
    def compute_cross_modal_similarity(
        self,
        video_features: torch.Tensor,
        audio_features: torch.Tensor,
    ) -> float:
        """Compute cross-modal similarity between video and audio features."""
        # Compute normalized cross-correlation
        video_norm = F.normalize(video_features, dim=-1)
        audio_norm = F.normalize(audio_features, dim=-1)
        
        similarity = torch.sum(video_norm * audio_norm, dim=-1)
        
        return similarity.mean().item()


def compute_comprehensive_metrics(
    pred_video: torch.Tensor,
    target_video: torch.Tensor,
    pred_audio: torch.Tensor,
    target_audio: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """
    Compute comprehensive multimodal metrics.
    
    Args:
        pred_video: Predicted video (B, T, C, H, W)
        target_video: Target video (B, T, C, H, W)
        pred_audio: Predicted audio (B, T, F)
        target_audio: Target audio (B, T, F)
        device: Device for computation
        
    Returns:
        Dictionary of all metrics
    """
    metrics_calculator = PerceptualMetrics(device)
    
    # Video metrics
    video_metrics = metrics_calculator.compute_video_metrics(pred_video, target_video)
    
    # Audio metrics
    audio_metrics = metrics_calculator.compute_audio_metrics(pred_audio, target_audio)
    
    # Combine metrics
    all_metrics = {}
    all_metrics.update({f'video_{k}': v for k, v in video_metrics.items()})
    all_metrics.update({f'audio_{k}': v for k, v in audio_metrics.items()})
    
    return all_metrics 