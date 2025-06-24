# OpenWorld-Multimodal Quick Start Guide

## Overview

Get up and running with OpenWorld-Multimodal in minutes! This guide will walk you through installation, basic usage, and your first multimodal world model.

**Author:** Nik Jois <nikjois@llamasearch.ai>  
**Version:** 2.0.0

## Installation

### Requirements

- Python 3.9 or higher
- PyTorch 2.1 or higher
- 8GB+ RAM (16GB+ recommended)
- CUDA-compatible GPU (optional but recommended)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/nikjois/openworld-multimodal.git
cd openworld-multimodal

# Install in development mode
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"

# Install UI dependencies (optional)
pip install -e ".[ui]"
```

### Verify Installation

```bash
python -c "import openworld; print('Installation successful!')"
```

## Quick Demo

Run the standalone demo to verify everything works:

```bash
python demo.py
```

Expected output:
```
OpenWorld-Multimodal Demo
==================================================
Author: Nik Jois <nikjois@llamasearch.ai>
Version: 2.0.0

System Information:
   PyTorch: 2.3.0
   Device: cuda

Creating OpenWorld-Multimodal Model...
   Model created in 3.01s
   Total Parameters: 611,460,814
   Trainable Parameters: 611,460,814
   Model Size: 2332.5 MB (FP32)

...

OpenWorld-Multimodal Demo Completed Successfully!
```

## Basic Usage

### 1. Create a Model

```python
import torch
from openworld.models.transformer_world_model import TransformerWorldModel

# Create a basic model
model = TransformerWorldModel(
    img_size=128,        # Smaller for quick testing
    patch_size=16,
    embed_dim=512,
    depth=6,             # Fewer layers for speed
    num_heads=8,
    use_physics_loss=True,
)

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

### 2. Prepare Data

```python
# Create sample video and audio data
batch_size = 2
sequence_length = 8

# Video: (batch, time, channels, height, width)
video = torch.randn(batch_size, sequence_length, 3, 128, 128).to(device)

# Audio: (batch, time, features)
audio = torch.randn(batch_size, sequence_length, 128).to(device)
```

### 3. Run Forward Pass

```python
# Set model to evaluation mode
model.eval()

with torch.no_grad():
    # Basic reconstruction
    outputs = model(video, audio)
    
    # Get reconstructed data
    reconstructed_video = outputs['reconstruction']['video']
    reconstructed_audio = outputs['reconstruction']['audio']
    
    print(f"Original video shape: {video.shape}")
    print(f"Reconstructed video shape: {reconstructed_video.shape}")
```

### 4. Future Prediction

```python
with torch.no_grad():
    # Predict 4 future frames
    outputs = model(video, audio, future_steps=4)
    
    future_video = outputs['future_video']
    future_audio = outputs['future_audio']
    
    print(f"Future video shape: {future_video.shape}")
    print(f"Future audio shape: {future_audio.shape}")
```

## Command Line Interface

### Quick Commands

```bash
# Get system information
python demo_cli.py info

# Run comprehensive demo
python demo_cli.py demo --verbose

# Run benchmarks
python demo_cli.py benchmark

# Interactive generation
python demo_cli.py generate --interactive
```

### Using the CLI Tool

```bash
# Train a model (requires dataset)
openworld train --config experiments/configs/default_config.json

# Evaluate a trained model
openworld evaluate --checkpoint model.pt --dataset synthetic

# Generate content
openworld generate --checkpoint model.pt --frames 64
```

## Working with Real Data

### Using HuggingFace Datasets

```python
from openworld.data.hf_datasets import create_dataloaders

# Create dataloaders for training
dataloaders = create_dataloaders(
    datasets='synthetic',  # Start with synthetic data
    batch_size=16,
    sequence_length=8,
    video_size=128,        # Smaller for quick testing
)

train_loader = dataloaders['train']
val_loader = dataloaders['val']

# Iterate through data
for batch in train_loader:
    video = batch['video']      # Shape: (B, T, C, H, W)
    audio = batch['audio']      # Shape: (B, T, F)
    metadata = batch['metadata']
    break  # Just check first batch
```

### Custom Data Loading

```python
# Create your own data
def create_sample_batch(batch_size=4, seq_len=8):
    """Create a sample batch for testing."""
    video = torch.randn(batch_size, seq_len, 3, 128, 128)
    audio = torch.randn(batch_size, seq_len, 128)
    
    return {
        'video': video,
        'audio': audio,
        'metadata': {'source': 'synthetic'}
    }

# Use custom data
batch = create_sample_batch()
outputs = model(batch['video'].to(device), batch['audio'].to(device))
```

## Training Your First Model

### Basic Training Setup

```python
from openworld.training.trainer import create_trainer
from openworld.training.losses import MultimodalLoss

# Create loss function
loss_fn = MultimodalLoss(
    reconstruction_weight=1.0,
    perceptual_weight=0.1,
    physics_weight=0.1,
)

# Create trainer
trainer = create_trainer(
    model=model,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    loss_fn=loss_fn,
    num_epochs=10,           # Start small
    learning_rate=1e-4,
    batch_size=16,
    mixed_precision='fp16',  # For faster training
)

# Start training
trainer.train()
```

### Monitor Training

```python
# Training with monitoring
trainer = create_trainer(
    model=model,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    num_epochs=10,
    checkpoint_dir='./checkpoints',
    save_every_n_epochs=2,
    early_stopping_patience=5,
    use_wandb=True,          # Enable Weights & Biases logging
    project_name='openworld-quickstart',
)

trainer.train()
```

## Evaluation and Metrics

### Quick Evaluation

```python
from openworld.evaluation.benchmarks import BenchmarkSuite

# Create benchmark suite
benchmark = BenchmarkSuite(
    model=model,
    device=device,
    metrics=['psnr', 'ssim', 'physics_consistency'],
)

# Run evaluation
results = benchmark.run_all_benchmarks(val_loader)

print("Evaluation Results:")
for metric, value in results.items():
    if isinstance(value, dict):
        for k, v in value.items():
            print(f"  {metric}/{k}: {v:.4f}")
    else:
        print(f"  {metric}: {value:.4f}")
```

### Custom Metrics

```python
from openworld.evaluation.perceptual_metrics import calculate_psnr, calculate_ssim

# Calculate metrics manually
with torch.no_grad():
    outputs = model(video, audio)
    reconstructed = outputs['reconstruction']['video']
    
    # Calculate PSNR and SSIM
    psnr = calculate_psnr(reconstructed, video)
    ssim = calculate_ssim(reconstructed, video)
    
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")
```

## Generation and Sampling

### Basic Generation

```python
from openworld.generation.sampler import VideoGenerator

# Create generator
generator = VideoGenerator(
    model=model,
    device=device,
    temperature=0.8,      # Control randomness
    top_p=0.9,           # Nucleus sampling
)

# Generate unconditional video
generated_video = generator.generate_unconditional(
    num_frames=32,
    video_size=(128, 128),
)

print(f"Generated video shape: {generated_video.shape}")
```

### Conditional Generation

```python
# Generate from a prompt
prompt_video = video[:1, :4]  # Use first 4 frames as prompt
prompt_audio = audio[:1, :4]

generated = generator.generate_from_prompt(
    prompt_video=prompt_video,
    prompt_audio=prompt_audio,
    num_frames=16,
)

print(f"Generated from prompt: {generated.shape}")
```

## Configuration

### Model Configuration

Create a configuration file `config.json`:

```json
{
  "model": {
    "img_size": 128,
    "patch_size": 16,
    "embed_dim": 512,
    "depth": 6,
    "num_heads": 8,
    "latent_dim": 256,
    "use_physics_loss": true
  },
  "training": {
    "batch_size": 16,
    "learning_rate": 1e-4,
    "num_epochs": 50,
    "mixed_precision": "fp16"
  },
  "data": {
    "dataset": "synthetic",
    "sequence_length": 8,
    "video_size": 128
  }
}
```

Load and use configuration:

```python
import json

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Create model from config
model = TransformerWorldModel(**config['model'])

# Use training config
trainer = create_trainer(
    model=model,
    train_dataloader=train_loader,
    **config['training']
)
```

## Common Workflows

### Workflow 1: Quick Experimentation

```python
# 1. Create small model for quick testing
model = TransformerWorldModel(
    img_size=64,     # Very small
    embed_dim=256,   # Reduced size
    depth=4,         # Fewer layers
    num_heads=4,
)

# 2. Create synthetic data
batch = create_sample_batch(batch_size=2, seq_len=4)

# 3. Quick forward pass
outputs = model(batch['video'], batch['audio'])

# 4. Check outputs
print("Model works! Shapes:")
for key, value in outputs['reconstruction'].items():
    print(f"  {key}: {value.shape}")
```

### Workflow 2: Training from Scratch

```python
# 1. Create full-size model
model = TransformerWorldModel()

# 2. Create real dataloaders
dataloaders = create_dataloaders(
    datasets='kinetics',  # Real dataset
    batch_size=32,
    sequence_length=16,
)

# 3. Set up training
trainer = create_trainer(
    model=model,
    train_dataloader=dataloaders['train'],
    val_dataloader=dataloaders['val'],
    num_epochs=100,
    use_wandb=True,
)

# 4. Train
trainer.train()
```

### Workflow 3: Fine-tuning

```python
# 1. Load pre-trained model
checkpoint = torch.load('pretrained_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# 2. Create trainer with lower learning rate
trainer = create_trainer(
    model=model,
    train_dataloader=dataloaders['train'],
    learning_rate=1e-5,  # Lower for fine-tuning
    num_epochs=20,
)

# 3. Fine-tune
trainer.train()
```

## Troubleshooting

### Common Issues

#### Out of Memory
```python
# Reduce batch size
trainer = create_trainer(
    model=model,
    train_dataloader=train_loader,
    batch_size=8,        # Smaller batch
    gradient_accumulation_steps=4,  # Simulate larger batch
)
```

#### Slow Training
```python
# Enable optimizations
trainer = create_trainer(
    model=model,
    train_dataloader=train_loader,
    mixed_precision='fp16',  # Faster training
    compile_model=True,      # PyTorch 2.0 compilation
)
```

#### Poor Quality
```python
# Increase model capacity
model = TransformerWorldModel(
    embed_dim=768,      # Larger embedding
    depth=12,           # More layers
    use_physics_loss=True,  # Better dynamics
)
```

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Create trainer with debug info
trainer = create_trainer(
    model=model,
    train_dataloader=train_loader,
    debug=True,
    profile=True,  # Performance profiling
)
```

## Next Steps

### Advanced Features

1. **Multi-GPU Training**: Use `accelerate` for distributed training
2. **Custom Datasets**: Implement your own dataset classes
3. **Advanced Generation**: Explore beam search and guided generation
4. **Physics Constraints**: Add custom physics losses
5. **Evaluation Metrics**: Implement domain-specific metrics

### Resources

- **Documentation**: [Full API Reference](API_REFERENCE.md)
- **Architecture**: [Architecture Overview](ARCHITECTURE.md)
- **Examples**: Check the `examples/` directory
- **Community**: Join our discussions on GitHub

### Example Projects

1. **Video Prediction**: Predict future frames from video sequences
2. **Audio-Visual Generation**: Generate synchronized audio and video
3. **Physics Simulation**: Model realistic object interactions
4. **Creative Applications**: Generate artistic videos and animations

## Support

Need help? Here's how to get support:

1. **Check Documentation**: Start with this guide and the API reference
2. **Run Demos**: Use `demo.py` and `demo_cli.py` to verify installation
3. **GitHub Issues**: Report bugs and request features
4. **Email**: Contact nikjois@llamasearch.ai for technical questions

## Conclusion

You're now ready to start using OpenWorld-Multimodal! Begin with the quick demo, experiment with small models, and gradually scale up to your specific use case.

Happy modeling! 