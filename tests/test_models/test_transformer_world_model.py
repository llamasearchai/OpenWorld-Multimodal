"""
Tests for OpenWorld-Multimodal Transformer World Model

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

from openworld.models.transformer_world_model import TransformerWorldModel


class TestTransformerWorldModel:
    """Test suite for TransformerWorldModel."""
    
    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def model_config(self):
        """Get test model configuration."""
        return {
            'img_size': 64,
            'patch_size': 8,
            'in_channels': 3,
            'audio_dim': 128,
            'audio_seq_len': 32,
            'embed_dim': 256,
            'depth': 4,
            'num_heads': 4,
            'mlp_ratio': 2.0,
            'latent_dim': 128,
            'num_latent_tokens': 16,
            'decode_depth': 2,
        }
    
    @pytest.fixture
    def model(self, model_config, device):
        """Create test model."""
        model = TransformerWorldModel(**model_config)
        return model.to(device)
    
    @pytest.fixture
    def sample_data(self, device):
        """Create sample input data."""
        batch_size = 2
        seq_len = 4
        
        video = torch.randn(batch_size, seq_len, 3, 64, 64).to(device)
        audio = torch.randn(batch_size, seq_len, 128).to(device)
        timesteps = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to(device)
        
        return {
            'video': video,
            'audio': audio,
            'timesteps': timesteps,
        }
    
    def test_model_creation(self, model_config):
        """Test model can be created with given configuration."""
        model = TransformerWorldModel(**model_config)
        assert isinstance(model, TransformerWorldModel)
        
        # Check model has expected attributes
        assert hasattr(model, 'vision_embed')
        assert hasattr(model, 'audio_embed')
        assert hasattr(model, 'encoder_blocks')
        assert hasattr(model, 'dynamics_transformer')
        assert hasattr(model, 'decoder_blocks')
    
    def test_model_forward_basic(self, model, sample_data):
        """Test basic forward pass."""
        outputs = model(
            video=sample_data['video'],
            audio=sample_data['audio'],
        )
        
        assert isinstance(outputs, dict)
        assert 'reconstruction' in outputs
        
        # Check output shapes
        reconstruction = outputs['reconstruction']
        assert 'video' in reconstruction
        assert 'audio' in reconstruction
    
    def test_model_forward_with_future_steps(self, model, sample_data):
        """Test forward pass with future prediction."""
        future_steps = 3
        
        outputs = model(
            video=sample_data['video'],
            audio=sample_data['audio'],
            future_steps=future_steps,
        )
        
        assert isinstance(outputs, dict)
        assert 'reconstruction' in outputs
        
        if 'future_video' in outputs:
            assert outputs['future_video'].shape[1] == future_steps
        if 'future_audio' in outputs:
            assert outputs['future_audio'].shape[1] == future_steps
    
    def test_model_forward_with_intermediates(self, model, sample_data):
        """Test forward pass returning intermediates."""
        outputs = model(
            video=sample_data['video'],
            audio=sample_data['audio'],
            return_intermediates=True,
        )
        
        assert isinstance(outputs, dict)
        assert 'reconstruction' in outputs
        
        if 'intermediates' in outputs:
            assert isinstance(outputs['intermediates'], list)
            assert len(outputs['intermediates']) > 0
    
    def test_model_forward_with_timesteps(self, model, sample_data):
        """Test forward pass with timestep information."""
        outputs = model(
            video=sample_data['video'],
            audio=sample_data['audio'],
            timesteps=sample_data['timesteps'],
        )
        
        assert isinstance(outputs, dict)
        assert 'reconstruction' in outputs
    
    def test_vision_encoding(self, model, sample_data):
        """Test vision encoding functionality."""
        encoded = model.encode_vision(
            sample_data['video'],
            sample_data['timesteps'],
        )
        
        batch_size, seq_len = sample_data['video'].shape[:2]
        expected_seq_len = seq_len * model.num_patches + seq_len  # patches + CLS tokens
        
        assert encoded.shape[0] == batch_size
        assert encoded.shape[2] == model.embed_dim
    
    def test_audio_encoding(self, model, sample_data):
        """Test audio encoding functionality."""
        encoded = model.encode_audio(sample_data['audio'])
        
        batch_size, seq_len = sample_data['audio'].shape[:2]
        
        assert encoded.shape[0] == batch_size
        assert encoded.shape[1] == seq_len
        assert encoded.shape[2] == model.embed_dim
    
    def test_model_parameters_count(self, model):
        """Test model has reasonable number of parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Should have some parameters
        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params == total_params  # All parameters should be trainable by default
        
        # Should be reasonable size (not too large for test)
        assert total_params < 200_000_000  # Less than 200M parameters for test model
    
    def test_model_gradient_flow(self, model, sample_data):
        """Test gradients flow properly through the model."""
        model.train()
        
        outputs = model(
            video=sample_data['video'],
            audio=sample_data['audio'],
        )
        
        # Create dummy loss
        reconstruction = outputs['reconstruction']
        if 'video' in reconstruction:
            loss = reconstruction['video'].mean()
        else:
            loss = torch.tensor(1.0, requires_grad=True)
        
        loss.backward()
        
        # Check gradients exist
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None:
                has_gradients = True
                break
        
        assert has_gradients, "No gradients found in model parameters"
    
    def test_model_eval_mode(self, model, sample_data):
        """Test model works in evaluation mode."""
        model.eval()
        
        with torch.no_grad():
            outputs = model(
                video=sample_data['video'],
                audio=sample_data['audio'],
            )
        
        assert isinstance(outputs, dict)
        assert 'reconstruction' in outputs
    
    def test_model_different_batch_sizes(self, model_config, device):
        """Test model works with different batch sizes."""
        model = TransformerWorldModel(**model_config).to(device)
        
        for batch_size in [1, 3, 5]:
            video = torch.randn(batch_size, 4, 3, 64, 64).to(device)
            audio = torch.randn(batch_size, 4, 128).to(device)
            
            outputs = model(video=video, audio=audio)
            
            assert isinstance(outputs, dict)
            assert 'reconstruction' in outputs
    
    def test_model_different_sequence_lengths(self, model_config, device):
        """Test model works with different sequence lengths."""
        model = TransformerWorldModel(**model_config).to(device)
        
        for seq_len in [2, 6, 8]:
            video = torch.randn(2, seq_len, 3, 64, 64).to(device)
            audio = torch.randn(2, seq_len, 128).to(device)
            
            outputs = model(video=video, audio=audio)
            
            assert isinstance(outputs, dict)
            assert 'reconstruction' in outputs
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_cuda_compatibility(self, model_config):
        """Test model works on CUDA."""
        device = torch.device('cuda')
        model = TransformerWorldModel(**model_config).to(device)
        
        video = torch.randn(2, 4, 3, 64, 64).to(device)
        audio = torch.randn(2, 4, 128).to(device)
        
        outputs = model(video=video, audio=audio)
        
        assert isinstance(outputs, dict)
        assert 'reconstruction' in outputs
        
        # Check outputs are on correct device
        reconstruction = outputs['reconstruction']
        if 'video' in reconstruction:
            assert reconstruction['video'].device.type == 'cuda'
    
    def test_model_memory_efficiency(self, model, sample_data):
        """Test model doesn't consume excessive memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
        
        # Forward pass
        outputs = model(
            video=sample_data['video'],
            audio=sample_data['audio'],
        )
        
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated()
            memory_used = peak_memory - initial_memory
            
            # Should use reasonable amount of memory (less than 1GB for test model)
            assert memory_used < 1024 * 1024 * 1024  # 1GB
    
    def test_model_output_shapes_consistency(self, model, sample_data):
        """Test output shapes are consistent across multiple runs."""
        outputs1 = model(
            video=sample_data['video'],
            audio=sample_data['audio'],
            future_steps=2,
        )
        
        outputs2 = model(
            video=sample_data['video'],
            audio=sample_data['audio'],
            future_steps=2,
        )
        
        # Check shapes are consistent
        for key in outputs1:
            if key in outputs2:
                if isinstance(outputs1[key], torch.Tensor) and isinstance(outputs2[key], torch.Tensor):
                    assert outputs1[key].shape == outputs2[key].shape
                elif isinstance(outputs1[key], dict) and isinstance(outputs2[key], dict):
                    for subkey in outputs1[key]:
                        if subkey in outputs2[key]:
                            if isinstance(outputs1[key][subkey], torch.Tensor):
                                assert outputs1[key][subkey].shape == outputs2[key][subkey].shape


if __name__ == '__main__':
    pytest.main([__file__]) 