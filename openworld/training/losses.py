"""
Loss functions for OpenWorld-Multimodal training.

Author: Nik Jois <nikjois@llamasearch.ai>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import lpips


class MultimodalLoss(nn.Module):
    """Comprehensive loss function for multimodal world modeling."""
    
    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        temporal_weight: float = 0.5,
        physics_weight: float = 0.2,
        use_perceptual: bool = True,
    ):
        super().__init__()
        
        self.reconstruction_weight = reconstruction_weight
        self.perceptual_weight = perceptual_weight
        self.temporal_weight = temporal_weight
        self.physics_weight = physics_weight
        self.use_perceptual = use_perceptual
        
        # Perceptual loss (LPIPS)
        if use_perceptual:
            try:
                self.lpips_loss = lpips.LPIPS(net='alex')
                self.lpips_loss.eval()
            except:
                self.lpips_loss = None
                self.use_perceptual = False
        
    def forward(
        self, 
        predictions: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multimodal loss.
        
        Args:
            predictions: Model predictions containing 'reconstruction' and optional 'future_*'
            targets: Ground truth data
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # Reconstruction loss
        if 'reconstruction' in predictions:
            recon_loss = self._reconstruction_loss(
                predictions['reconstruction'], 
                targets
            )
            losses.update(recon_loss)
        
        # Future prediction loss
        if 'future_video' in predictions:
            future_loss = self._future_prediction_loss(
                predictions['future_video'],
                predictions.get('future_audio'),
                targets
            )
            losses.update(future_loss)
            
        # Physics loss
        if 'physics' in predictions:
            physics_loss = self._physics_loss(predictions['physics'])
            losses['physics_loss'] = physics_loss * self.physics_weight
            
        # Temporal consistency loss
        if 'future_video' in predictions:
            temporal_loss = self._temporal_consistency_loss(
                predictions['reconstruction']['video'],
                predictions['future_video']
            )
            losses['temporal_loss'] = temporal_loss * self.temporal_weight
            
        # Compute total loss
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return losses
    
    def _reconstruction_loss(
        self, 
        reconstruction: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute reconstruction losses for each modality."""
        losses = {}
        
        # Video reconstruction loss
        if 'video' in reconstruction and 'video' in targets:
            video_pred = reconstruction['video']
            video_target = targets['video']
            
            # MSE loss
            mse_loss = F.mse_loss(video_pred, video_target)
            losses['video_mse'] = mse_loss * self.reconstruction_weight
            
            # Perceptual loss
            if self.use_perceptual and self.lpips_loss is not None:
                # Flatten batch and time dimensions for LPIPS
                B, T, C, H, W = video_pred.shape
                video_pred_flat = video_pred.view(B * T, C, H, W)
                video_target_flat = video_target.view(B * T, C, H, W)
                
                # Normalize to [-1, 1] for LPIPS
                video_pred_norm = 2 * video_pred_flat - 1
                video_target_norm = 2 * video_target_flat - 1
                
                perceptual_loss = self.lpips_loss(
                    video_pred_norm, video_target_norm
                ).mean()
                losses['video_perceptual'] = perceptual_loss * self.perceptual_weight
        
        # Audio reconstruction loss
        if 'audio' in reconstruction and 'audio' in targets:
            audio_pred = reconstruction['audio']
            audio_target = targets['audio']
            
            # MSE loss for audio
            audio_mse = F.mse_loss(audio_pred, audio_target)
            losses['audio_mse'] = audio_mse * self.reconstruction_weight * 0.1
            
        return losses
    
    def _future_prediction_loss(
        self,
        future_video: torch.Tensor,
        future_audio: Optional[torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute future prediction losses."""
        losses = {}
        
        # For future prediction, we typically don't have ground truth
        # Instead, we can use consistency losses or self-supervised objectives
        
        # Video future prediction consistency
        if future_video is not None:
            # Temporal smoothness loss
            B, T, C, H, W = future_video.shape
            if T > 1:
                diff = future_video[:, 1:] - future_video[:, :-1]
                smoothness_loss = torch.mean(torch.abs(diff))
                losses['future_smoothness'] = smoothness_loss * 0.1
        
        # Audio future prediction consistency
        if future_audio is not None:
            B, T, F = future_audio.shape
            if T > 1:
                diff = future_audio[:, 1:] - future_audio[:, :-1]
                audio_smoothness = torch.mean(torch.abs(diff))
                losses['future_audio_smoothness'] = audio_smoothness * 0.05
                
        return losses
    
    def _physics_loss(self, physics_predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute physics-informed losses."""
        physics_loss = 0.0
        
        if 'position' in physics_predictions and 'velocity' in physics_predictions:
            position = physics_predictions['position']
            velocity = physics_predictions['velocity']
            
            # Energy conservation (simple kinetic energy)
            kinetic_energy = 0.5 * torch.sum(velocity ** 2, dim=-1)
            energy_variance = torch.var(kinetic_energy)
            physics_loss += energy_variance
            
            # Momentum conservation (center of mass should be stable)
            momentum = velocity.mean(dim=0)
            momentum_magnitude = torch.norm(momentum)
            physics_loss += momentum_magnitude
            
        return physics_loss
    
    def _temporal_consistency_loss(
        self, 
        current_frames: torch.Tensor,
        future_frames: torch.Tensor
    ) -> torch.Tensor:
        """Compute temporal consistency between current and future frames."""
        # Take the last frame of current and first frame of future
        last_current = current_frames[:, -1]  # (B, C, H, W)
        first_future = future_frames[:, 0]    # (B, C, H, W)
        
        # They should be similar (temporal continuity)
        consistency_loss = F.mse_loss(last_current, first_future)
        
        return consistency_loss


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features."""
    
    def __init__(self, layers: list = ['relu_1_2', 'relu_2_2', 'relu_3_2']):
        super().__init__()
        
        from torchvision.models import vgg16
        vgg = vgg16(pretrained=True).features
        self.layers = layers
        
        # Extract specific layers
        self.feature_extractor = nn.ModuleDict()
        layer_names = ['relu_1_2', 'relu_2_2', 'relu_3_2', 'relu_4_2']
        layer_indices = [4, 9, 16, 23]
        
        for name, idx in zip(layer_names, layer_indices):
            if name in layers:
                self.feature_extractor[name] = nn.Sequential(*list(vgg.children())[:idx+1])
                
        # Freeze parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss."""
        loss = 0.0
        
        for layer_name, extractor in self.feature_extractor.items():
            pred_features = extractor(pred)
            target_features = extractor(target)
            loss += F.mse_loss(pred_features, target_features)
            
        return loss / len(self.feature_extractor)


class GradientPenalty(nn.Module):
    """Gradient penalty for training stability."""
    
    def __init__(self, lambda_gp: float = 10.0):
        super().__init__()
        self.lambda_gp = lambda_gp
        
    def forward(self, real_data: torch.Tensor, fake_data: torch.Tensor, critic: nn.Module) -> torch.Tensor:
        """Compute gradient penalty."""
        batch_size = real_data.size(0)
        
        # Random interpolation between real and fake data
        alpha = torch.rand(batch_size, 1, 1, 1, device=real_data.device)
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        
        # Critic output for interpolated data
        critic_output = critic(interpolated)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=critic_output,
            inputs=interpolated,
            grad_outputs=torch.ones_like(critic_output),
            create_graph=True,
            retain_graph=True,
        )[0]
        
        # Gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()
        
        return self.lambda_gp * penalty


class SpectralLoss(nn.Module):
    """Spectral loss for audio modeling."""
    
    def __init__(self, n_fft: int = 2048, hop_length: int = 512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute spectral loss."""
        # Convert to spectrograms
        pred_spec = torch.stft(
            pred.flatten(0, 1), 
            n_fft=self.n_fft, 
            hop_length=self.hop_length,
            return_complex=True
        )
        target_spec = torch.stft(
            target.flatten(0, 1), 
            n_fft=self.n_fft, 
            hop_length=self.hop_length,
            return_complex=True
        )
        
        # Magnitude and phase losses
        pred_mag = torch.abs(pred_spec)
        target_mag = torch.abs(target_spec)
        mag_loss = F.mse_loss(pred_mag, target_mag)
        
        pred_phase = torch.angle(pred_spec)
        target_phase = torch.angle(target_spec)
        phase_loss = F.mse_loss(pred_phase, target_phase)
        
        return mag_loss + 0.1 * phase_loss 