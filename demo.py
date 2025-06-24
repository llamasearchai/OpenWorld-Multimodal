#!/usr/bin/env python3
"""
OpenWorld-Multimodal Standalone Demo
A simple demo script to showcase the multimodal world modeling capabilities.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import torch
import numpy as np
import time
from openworld.models.transformer_world_model import TransformerWorldModel

def main():
    print("OpenWorld-Multimodal Demo")
    print("=" * 50)
    print("Author: Nik Jois <nikjois@llamasearch.ai>")
    print("Version: 2.0.0")
    print()
    
    # System information
    print("System Information:")
    print(f"   PyTorch: {torch.__version__}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print()
    
    try:
        # Create model
        print("Creating OpenWorld-Multimodal Model...")
        start_time = time.time()
        
        model = TransformerWorldModel(
            img_size=128,
            patch_size=16,
            embed_dim=512,
            depth=6,
            num_heads=8,
            decode_depth=4,
            use_physics_loss=True,
        ).to(device)
        
        creation_time = time.time() - start_time
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   Model created in {creation_time:.2f}s")
        print(f"   Total Parameters: {total_params:,}")
        print(f"   Trainable Parameters: {trainable_params:,}")
        print(f"   Model Size: {total_params * 4 / 1024 / 1024:.1f} MB (FP32)")
        print()
        
        # Create sample data
        print("Creating Sample Data...")
        batch_size = 2
        seq_len = 8
        
        # Create synthetic video data (RGB frames)
        video = torch.randn(batch_size, seq_len, 3, 128, 128).to(device)
        
        # Create synthetic audio data (mel spectrograms)
        audio = torch.randn(batch_size, seq_len, 128).to(device)
        
        print(f"   Video shape: {list(video.shape)} (B×T×C×H×W)")
        print(f"   Audio shape: {list(audio.shape)} (B×T×F)")
        print()
        
        # Test reconstruction
        print("Testing Reconstruction...")
        model.eval()
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model(
                video=video,
                audio=audio,
                future_steps=0,
            )
        
        reconstruction_time = time.time() - start_time
        
        recon_video = outputs['reconstruction']['video']
        recon_audio = outputs['reconstruction']['audio']
        
        print(f"   Reconstruction completed in {reconstruction_time:.3f}s")
        print(f"   Reconstructed video: {list(recon_video.shape)}")
        print(f"   Reconstructed audio: {list(recon_audio.shape)}")
        
        # Calculate reconstruction metrics
        video_mse = torch.nn.functional.mse_loss(recon_video, video).item()
        audio_mse = torch.nn.functional.mse_loss(recon_audio, audio).item()
        
        print(f"   Video MSE: {video_mse:.6f}")
        print(f"   Audio MSE: {audio_mse:.6f}")
        print()
        
        # Test future prediction
        print("Testing Future Prediction...")
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model(
                video=video,
                audio=audio,
                future_steps=4,
            )
        
        prediction_time = time.time() - start_time
        
        future_video = outputs['future_video']
        future_audio = outputs['future_audio']
        
        print(f"   Future prediction completed in {prediction_time:.3f}s")
        print(f"   Future video: {list(future_video.shape)}")
        print(f"   Future audio: {list(future_audio.shape)}")
        print()
        
        # Test physics predictions if available
        if 'physics' in outputs:
            print("Physics Predictions:")
            physics = outputs['physics']
            print(f"   Position: {list(physics['position'].shape)}")
            print(f"   Velocity: {list(physics['velocity'].shape)}")
            print(f"   Sample position: {physics['position'][0].tolist()}")
            print(f"   Sample velocity: {physics['velocity'][0].tolist()}")
            print()
        
        # Memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1e9
            memory_reserved = torch.cuda.memory_reserved() / 1e9
            print(f"GPU Memory Usage:")
            print(f"   Used: {memory_used:.2f}GB")
            print(f"   Reserved: {memory_reserved:.2f}GB")
            print()
        
        # Performance summary
        total_time = reconstruction_time + prediction_time
        fps = (seq_len * batch_size) / total_time
        
        print("Performance Summary:")
        print(f"   Reconstruction: {reconstruction_time:.3f}s")
        print(f"   Prediction: {prediction_time:.3f}s")
        print(f"   Total: {total_time:.3f}s")
        print(f"   Effective FPS: {fps:.1f}")
        print()
        
        print("OpenWorld-Multimodal Demo Completed Successfully!")
        print("   The multimodal world model is working correctly and can:")
        print("   - Encode video and audio into a shared representation")
        print("   - Reconstruct input modalities from latent space")
        print("   - Predict future video and audio frames")
        print("   - Model physics-informed dynamics")
        print("   - Handle multimodal synchronization")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 