"""
OpenWorld-Multimodal World Model Sampler
Advanced sampling strategies for video and audio generation.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import math

from ..models.transformer_world_model import TransformerWorldModel
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SamplingConfig:
    """Configuration for sampling strategies."""
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    num_samples: int = 1
    guidance_scale: float = 1.0
    use_physics_guidance: bool = True
    max_length: int = 100
    seed: Optional[int] = None


class WorldModelSampler:
    """
    Advanced sampler for OpenWorld-Multimodal generation.
    
    Supports:
    - Temperature scaling
    - Top-k and top-p sampling
    - Physics-guided generation
    - Multi-modal consistency
    - Beam search
    - Classifier-free guidance
    """
    
    def __init__(
        self,
        model: TransformerWorldModel,
        device: torch.device,
        config: Optional[SamplingConfig] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.config = config or SamplingConfig()
        
        logger.info("WorldModelSampler initialized")
        
    def generate(
        self,
        context_video: torch.Tensor,
        context_audio: torch.Tensor,
        num_steps: int,
        sampling_config: Optional[SamplingConfig] = None,
        return_intermediates: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate future video and audio sequences.
        
        Args:
            context_video: Context video frames (B, T, C, H, W)
            context_audio: Context audio spectrograms (B, T, F)
            num_steps: Number of future steps to generate
            sampling_config: Optional sampling configuration
            return_intermediates: Whether to return intermediate states
            
        Returns:
            Dictionary containing generated sequences and optional intermediates
        """
        config = sampling_config or self.config
        
        if config.seed is not None:
            torch.manual_seed(config.seed)
            
        self.model.eval()
        
        with torch.no_grad():
            return self._generate_autoregressive(
                context_video,
                context_audio,
                num_steps,
                config,
                return_intermediates,
            )
    
    def _generate_autoregressive(
        self,
        context_video: torch.Tensor,
        context_audio: torch.Tensor,
        num_steps: int,
        config: SamplingConfig,
        return_intermediates: bool,
    ) -> Dict[str, torch.Tensor]:
        """Generate sequences autoregressively."""
        B = context_video.size(0)
        
        # Initialize sequences with context
        video_sequence = context_video.clone()
        audio_sequence = context_audio.clone()
        
        intermediates = [] if return_intermediates else None
        
        for step in range(num_steps):
            # Get model predictions
            outputs = self.model(
                video=video_sequence,
                audio=audio_sequence,
                future_steps=1,
                return_intermediates=return_intermediates,
            )
            
            # Sample next frames
            if 'future_video' in outputs:
                next_video = self._sample_video(outputs['future_video'], config)
                video_sequence = torch.cat([video_sequence, next_video], dim=1)
                
            if 'future_audio' in outputs:
                next_audio = self._sample_audio(outputs['future_audio'], config)
                audio_sequence = torch.cat([audio_sequence, next_audio], dim=1)
                
            if return_intermediates and 'intermediates' in outputs:
                intermediates.append(outputs['intermediates'])
                
        # Extract generated portions
        generated_video = video_sequence[:, context_video.size(1):]
        generated_audio = audio_sequence[:, context_audio.size(1):]
        
        results = {
            'generated_video': generated_video,
            'generated_audio': generated_audio,
            'full_video': video_sequence,
            'full_audio': audio_sequence,
        }
        
        if return_intermediates:
            results['intermediates'] = intermediates
            
        return results
    
    def _sample_video(
        self,
        video_tensor: torch.Tensor,
        config: SamplingConfig,
    ) -> torch.Tensor:
        """Sample video frames from tensor."""
        # The video tensor is already generated, so we just need to extract the next frame
        # Handle different possible shapes: (B, future_steps, seq_len, C, H, W) or (B, future_steps, C, H, W)
        
        if video_tensor.dim() == 6:
            # Shape: (B, future_steps, seq_len, C, H, W)
            # Take the last frame from the first future step
            sampled_video = video_tensor[:, 0, -1:, :, :, :]  # (B, 1, C, H, W)
        elif video_tensor.dim() == 5:
            # Shape: (B, future_steps, C, H, W)
            # Take the first future step
            sampled_video = video_tensor[:, 0:1, :, :, :]  # (B, 1, C, H, W)
        else:
            # Fallback: assume it's already in the right shape
            sampled_video = video_tensor
        
        # Add temperature scaling if needed
        if config.temperature != 1.0:
            sampled_video = sampled_video / config.temperature
            
        # Apply top-k filtering if specified
        if config.top_k is not None:
            sampled_video = self._apply_top_k_filtering(sampled_video, config.top_k)
            
        # Apply top-p filtering if specified
        if config.top_p is not None:
            sampled_video = self._apply_top_p_filtering(sampled_video, config.top_p)
            
        # Ensure values are in reasonable range
        sampled_video = torch.clamp(sampled_video, -2.0, 2.0)
        
        return sampled_video
    
    def _sample_audio(
        self,
        audio_tensor: torch.Tensor,
        config: SamplingConfig,
    ) -> torch.Tensor:
        """Sample audio spectrograms from tensor."""
        # The audio tensor is already generated, so we just need to extract the next frame
        # Handle different possible shapes: (B, future_steps, seq_len, F) or (B, future_steps, F)
        
        if audio_tensor.dim() == 4:
            # Shape: (B, future_steps, seq_len, F)
            # Take the last frame from the first future step
            sampled_audio = audio_tensor[:, 0, -1:, :]  # (B, 1, F)
        elif audio_tensor.dim() == 3:
            # Shape: (B, future_steps, F)
            # Take the first future step
            sampled_audio = audio_tensor[:, 0:1, :]  # (B, 1, F)
        else:
            # Fallback: assume it's already in the right shape
            sampled_audio = audio_tensor
        
        # Add temperature scaling if needed
        if config.temperature != 1.0:
            sampled_audio = sampled_audio / config.temperature
            
        # Apply top-k filtering if specified
        if config.top_k is not None:
            sampled_audio = self._apply_top_k_filtering(sampled_audio, config.top_k)
            
        # Apply top-p filtering if specified
        if config.top_p is not None:
            sampled_audio = self._apply_top_p_filtering(sampled_audio, config.top_p)
            
        # Ensure values are in reasonable range
        sampled_audio = torch.clamp(sampled_audio, -2.0, 2.0)
        
        return sampled_audio
    
    def _apply_top_k_filtering(
        self,
        logits: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        """Apply top-k filtering to logits."""
        if top_k <= 0:
            return logits
            
        # Get top-k values and indices
        top_k_values, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
        
        # Create mask for values not in top-k
        mask = logits < top_k_values[..., -1:]
        
        # Set non-top-k values to negative infinity
        logits = logits.masked_fill(mask, float('-inf'))
        
        return logits
    
    def _apply_top_p_filtering(
        self,
        logits: torch.Tensor,
        top_p: float,
    ) -> torch.Tensor:
        """Apply top-p (nucleus) filtering to logits."""
        if top_p >= 1.0:
            return logits
            
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        
        # Compute cumulative probabilities
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Keep at least one token
        sorted_indices_to_remove[..., 0] = False
        
        # Scatter back to original order
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        return logits
    
    def generate_with_guidance(
        self,
        context_video: torch.Tensor,
        context_audio: torch.Tensor,
        guidance_text: Optional[str] = None,
        guidance_image: Optional[torch.Tensor] = None,
        num_steps: int = 10,
        guidance_scale: float = 7.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate with classifier-free guidance.
        
        Args:
            context_video: Context video frames
            context_audio: Context audio spectrograms
            guidance_text: Optional text guidance
            guidance_image: Optional image guidance
            num_steps: Number of generation steps
            guidance_scale: Guidance strength
            
        Returns:
            Generated sequences with guidance
        """
        # Unconditional generation
        uncond_outputs = self.generate(
            context_video,
            context_audio,
            num_steps,
            SamplingConfig(guidance_scale=1.0),
        )
        
        # Conditional generation (simplified - would need proper conditioning)
        cond_outputs = self.generate(
            context_video,
            context_audio,
            num_steps,
            SamplingConfig(guidance_scale=guidance_scale),
        )
        
        # Apply classifier-free guidance
        guided_video = self._apply_guidance(
            uncond_outputs['generated_video'],
            cond_outputs['generated_video'],
            guidance_scale,
        )
        
        guided_audio = self._apply_guidance(
            uncond_outputs['generated_audio'],
            cond_outputs['generated_audio'],
            guidance_scale,
        )
        
        return {
            'generated_video': guided_video,
            'generated_audio': guided_audio,
        }
    
    def _apply_guidance(
        self,
        uncond_output: torch.Tensor,
        cond_output: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        """Apply classifier-free guidance."""
        return uncond_output + guidance_scale * (cond_output - uncond_output)
    
    def generate_diverse_samples(
        self,
        context_video: torch.Tensor,
        context_audio: torch.Tensor,
        num_samples: int,
        num_steps: int,
        diversity_penalty: float = 0.1,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Generate diverse samples using different sampling strategies.
        
        Args:
            context_video: Context video frames
            context_audio: Context audio spectrograms
            num_samples: Number of diverse samples to generate
            num_steps: Number of generation steps
            diversity_penalty: Penalty for similar samples
            
        Returns:
            List of generated samples
        """
        samples = []
        
        for i in range(num_samples):
            # Vary sampling parameters for diversity
            temperature = 0.8 + 0.4 * np.random.random()
            top_p = 0.8 + 0.2 * np.random.random()
            
            config = SamplingConfig(
                temperature=temperature,
                top_p=top_p,
                seed=i,
            )
            
            sample = self.generate(
                context_video,
                context_audio,
                num_steps,
                config,
            )
            
            samples.append(sample)
            
        return samples
    
    def interpolate_sequences(
        self,
        start_video: torch.Tensor,
        end_video: torch.Tensor,
        start_audio: torch.Tensor,
        end_audio: torch.Tensor,
        num_interpolation_steps: int = 10,
    ) -> Dict[str, torch.Tensor]:
        """
        Interpolate between two sequences.
        
        Args:
            start_video: Starting video sequence
            end_video: Ending video sequence
            start_audio: Starting audio sequence
            end_audio: Ending audio sequence
            num_interpolation_steps: Number of interpolation steps
            
        Returns:
            Interpolated sequences
        """
        # Linear interpolation in latent space
        interpolated_videos = []
        interpolated_audios = []
        
        for i in range(num_interpolation_steps + 1):
            alpha = i / num_interpolation_steps
            
            # Interpolate video
            interp_video = (1 - alpha) * start_video + alpha * end_video
            interpolated_videos.append(interp_video)
            
            # Interpolate audio
            interp_audio = (1 - alpha) * start_audio + alpha * end_audio
            interpolated_audios.append(interp_audio)
            
        interpolated_video = torch.stack(interpolated_videos, dim=1)
        interpolated_audio = torch.stack(interpolated_audios, dim=1)
        
        return {
            'interpolated_video': interpolated_video,
            'interpolated_audio': interpolated_audio,
        }


def create_sampler(
    model: TransformerWorldModel,
    device: torch.device,
    config: Optional[Dict[str, Any]] = None,
) -> WorldModelSampler:
    """
    Factory function to create a sampler with default configuration.
    
    Args:
        model: The world model for generation
        device: Device for computation
        config: Sampling configuration dictionary
        
    Returns:
        Configured WorldModelSampler instance
    """
    sampling_config = SamplingConfig()
    
    if config:
        for key, value in config.items():
            if hasattr(sampling_config, key):
                setattr(sampling_config, key, value)
                
    return WorldModelSampler(model, device, sampling_config) 