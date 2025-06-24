# OpenWorld-Multimodal API Reference

## Overview

This document provides comprehensive API reference for the OpenWorld-Multimodal library.

**Author:** Nik Jois <nikjois@llamasearch.ai>  
**Version:** 2.0.0

## Core Models

### `openworld.models.TransformerWorldModel`

The main transformer-based world model for multimodal prediction.

```python
class TransformerWorldModel(nn.Module):
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        audio_dim: int = 128,
        audio_seq_len: int = 256,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        latent_dim: int = 512,
        num_latent_tokens: int = 32,
        decode_depth: int = 8,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        use_physics_loss: bool = True,
        physics_embed_dim: int = 128,
    )
```

#### Parameters

- **img_size** (int): Input image size. Default: 256
- **patch_size** (int): Vision transformer patch size. Default: 16
- **in_channels** (int): Number of input image channels. Default: 3
- **audio_dim** (int): Audio feature dimension. Default: 128
- **audio_seq_len** (int): Maximum audio sequence length. Default: 256
- **embed_dim** (int): Embedding dimension. Default: 768
- **depth** (int): Number of transformer encoder layers. Default: 12
- **num_heads** (int): Number of attention heads. Default: 12
- **mlp_ratio** (float): MLP expansion ratio. Default: 4.0
- **latent_dim** (int): Latent space dimension. Default: 512
- **num_latent_tokens** (int): Number of latent tokens. Default: 32
- **decode_depth** (int): Number of decoder layers. Default: 8
- **drop_rate** (float): Dropout rate. Default: 0.0
- **attn_drop_rate** (float): Attention dropout rate. Default: 0.0
- **use_physics_loss** (bool): Enable physics constraints. Default: True
- **physics_embed_dim** (int): Physics embedding dimension. Default: 128

#### Methods

##### `forward(video, audio, future_steps=0, timesteps=None, return_intermediates=False)`

Forward pass through the world model.

**Parameters:**
- **video** (torch.Tensor): Video tensor of shape (B, T, C, H, W)
- **audio** (torch.Tensor): Audio tensor of shape (B, T, F)
- **future_steps** (int): Number of future steps to predict. Default: 0
- **timesteps** (torch.Tensor, optional): Timestep information
- **return_intermediates** (bool): Return intermediate activations. Default: False

**Returns:**
- **dict**: Dictionary containing predictions with keys:
  - `reconstruction`: Reconstructed video and audio
  - `future_video`: Future video predictions (if future_steps > 0)
  - `future_audio`: Future audio predictions (if future_steps > 0)
  - `physics`: Physics predictions (if use_physics_loss=True)
  - `intermediates`: Intermediate activations (if return_intermediates=True)

##### `encode_vision(x, timesteps=None)`

Encode video frames into patch embeddings.

**Parameters:**
- **x** (torch.Tensor): Video tensor of shape (B, T, C, H, W)
- **timesteps** (torch.Tensor, optional): Timestep information

**Returns:**
- **torch.Tensor**: Encoded features of shape (B, T*N, D)

##### `encode_audio(audio)`

Encode audio spectrograms.

**Parameters:**
- **audio** (torch.Tensor): Audio tensor of shape (B, T, F)

**Returns:**
- **torch.Tensor**: Encoded features of shape (B, T, D)

#### Example Usage

```python
import torch
from openworld.models.transformer_world_model import TransformerWorldModel

# Create model
model = TransformerWorldModel(
    img_size=128,
    embed_dim=512,
    depth=6,
    num_heads=8,
)

# Create sample data
video = torch.randn(2, 8, 3, 128, 128)
audio = torch.randn(2, 8, 128)

# Forward pass
outputs = model(video, audio, future_steps=4)

# Access results
reconstruction = outputs['reconstruction']
future_video = outputs['future_video']
```

## Model Components

### `openworld.models.components.MultiheadAttention`

Efficient multi-head attention implementation.

```python
class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
    )
```

### `openworld.models.components.CrossModalAttention`

Cross-modal attention for fusion.

```python
class CrossModalAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        temperature: float = 1.0,
    )
```

### `openworld.models.components.SpatioTemporalEncoding`

Spatiotemporal positional encoding.

```python
class SpatioTemporalEncoding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        max_seq_len: int = 1000,
        max_spatial_size: int = 256,
        learnable: bool = True,
    )
```

## Data Loading

### `openworld.data.create_dataloaders`

Create train/validation/test dataloaders from HuggingFace datasets.

```python
def create_dataloaders(
    datasets: Union[str, List[str], List[Dict]],
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    video_size: int = 256,
    sequence_length: int = 16,
    audio_features: int = 128,
    augmentation: bool = True,
    **kwargs
) -> Dict[str, DataLoader]
```

**Parameters:**
- **datasets**: Dataset name(s) or configurations
- **batch_size**: Batch size for dataloaders
- **num_workers**: Number of data loading workers
- **pin_memory**: Whether to pin memory for GPU transfer
- **video_size**: Size to resize videos to
- **sequence_length**: Number of frames per sequence
- **audio_features**: Number of audio features (mel bins)
- **augmentation**: Whether to apply augmentations

**Returns:**
- **Dict[str, DataLoader]**: Dictionary containing train/val/test dataloaders

#### Example Usage

```python
from openworld.data.hf_datasets import create_dataloaders

# Create dataloaders for multiple datasets
dataloaders = create_dataloaders(
    datasets=['kinetics', 'something_something'],
    batch_size=32,
    sequence_length=16,
    video_size=256,
)

train_loader = dataloaders['train']
val_loader = dataloaders['val']
```

## Training

### `openworld.training.NeuralWorldTrainer`

Advanced training pipeline with distributed training support.

```python
class NeuralWorldTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        loss_fn: Optional[nn.Module] = None,
        num_epochs: int = 100,
        gradient_accumulation_steps: int = 1,
        mixed_precision: str = 'fp16',
        checkpoint_dir: str = './checkpoints',
        early_stopping_patience: Optional[int] = 10,
        use_ema: bool = True,
        **kwargs
    )
```

#### Methods

##### `train()`

Main training loop.

##### `load_checkpoint(checkpoint_path)`

Load model checkpoint.

##### `save_checkpoint(epoch, metrics)`

Save model checkpoint.

#### Example Usage

```python
from openworld.training.trainer import NeuralWorldTrainer

trainer = NeuralWorldTrainer(
    model=model,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    num_epochs=100,
    mixed_precision='fp16',
    use_ema=True,
)

trainer.train()
```

### `openworld.training.MultimodalLoss`

Multi-task loss function for world modeling.

```python
class MultimodalLoss(nn.Module):
    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        temporal_weight: float = 0.05,
        physics_weight: float = 0.1,
        spectral_weight: float = 0.05,
    )
```

## Evaluation

### `openworld.evaluation.BenchmarkSuite`

Comprehensive benchmark suite for evaluation.

```python
class BenchmarkSuite:
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = 'cuda',
        metrics: List[str] = None,
        save_predictions: bool = False,
        output_dir: str = './results',
    )
```

#### Methods

##### `run_all_benchmarks(dataloader)`

Run complete benchmark suite.

##### `benchmark_video_quality(dataloader)`

Benchmark video generation quality.

##### `benchmark_physics_consistency(dataloader)`

Benchmark physics understanding.

#### Example Usage

```python
from openworld.evaluation.benchmarks import BenchmarkSuite

benchmark = BenchmarkSuite(
    model=model,
    device='cuda',
    metrics=['psnr', 'ssim', 'lpips', 'fvd'],
)

results = benchmark.run_all_benchmarks(test_loader)
```

## Generation

### `openworld.generation.VideoGenerator`

Advanced video generation with multiple sampling strategies.

```python
class VideoGenerator:
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = 'cuda',
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        beam_size: int = 1,
        max_length: int = 100,
    )
```

#### Methods

##### `generate_unconditional(num_frames, **kwargs)`

Generate video without conditioning.

##### `generate_from_prompt(prompt_video, num_frames, **kwargs)`

Generate video from video prompt.

##### `generate_with_physics(initial_state, num_frames, **kwargs)`

Generate video with physics constraints.

#### Example Usage

```python
from openworld.generation.sampler import VideoGenerator

generator = VideoGenerator(
    model=model,
    temperature=0.8,
    top_p=0.9,
)

# Generate unconditional video
video = generator.generate_unconditional(num_frames=64)

# Generate from prompt
prompt = torch.randn(1, 8, 3, 256, 256)
video = generator.generate_from_prompt(prompt, num_frames=32)
```

## Command Line Interface

### Training

```bash
openworld train \
    --config config.json \
    --dataset kinetics \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --mixed-precision fp16
```

### Evaluation

```bash
openworld evaluate \
    --checkpoint model.pt \
    --dataset kinetics \
    --metrics psnr ssim lpips \
    --output-dir results/
```

### Generation

```bash
openworld generate \
    --checkpoint model.pt \
    --prompt video.mp4 \
    --output generated.mp4 \
    --frames 100 \
    --temperature 0.8
```

## Configuration

### Model Configuration

```json
{
  "model": {
    "img_size": 256,
    "patch_size": 16,
    "embed_dim": 768,
    "depth": 12,
    "num_heads": 12,
    "latent_dim": 512,
    "use_physics_loss": true
  },
  "training": {
    "batch_size": 32,
    "learning_rate": 1e-4,
    "num_epochs": 100,
    "mixed_precision": "fp16"
  },
  "data": {
    "datasets": ["kinetics", "something_something"],
    "sequence_length": 16,
    "video_size": 256,
    "augmentation": true
  }
}
```

## Error Handling

### Common Exceptions

#### `ModelConfigurationError`
Raised when model configuration is invalid.

#### `DataLoadingError`
Raised when data loading fails.

#### `TrainingError`
Raised during training failures.

### Example Error Handling

```python
try:
    model = TransformerWorldModel(**config)
    trainer = NeuralWorldTrainer(model, train_loader)
    trainer.train()
except ModelConfigurationError as e:
    print(f"Model configuration error: {e}")
except TrainingError as e:
    print(f"Training failed: {e}")
```

## Performance Tips

### Memory Optimization

1. **Use mixed precision training**: Set `mixed_precision='fp16'`
2. **Gradient accumulation**: Increase `gradient_accumulation_steps`
3. **Batch size tuning**: Start with smaller batches and increase gradually
4. **Sequence length**: Use shorter sequences for initial training

### Speed Optimization

1. **Data loading**: Increase `num_workers` for faster data loading
2. **Pin memory**: Set `pin_memory=True` for GPU training
3. **Compilation**: Use `torch.compile()` for PyTorch 2.0+
4. **Caching**: Enable attention caching for inference

### Quality Optimization

1. **EMA**: Use exponential moving averages for stable generation
2. **Physics loss**: Enable physics constraints for realistic dynamics
3. **Perceptual loss**: Include perceptual losses for better visual quality
4. **Curriculum learning**: Start with simple sequences and increase complexity

## Troubleshooting

### Common Issues

#### CUDA Out of Memory
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision
- Use smaller model variants

#### Slow Training
- Increase number of data loading workers
- Use faster storage (SSD)
- Enable mixed precision training
- Use distributed training for multiple GPUs

#### Poor Generation Quality
- Increase model size
- Use more training data
- Enable physics losses
- Tune generation hyperparameters

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug mode
trainer = NeuralWorldTrainer(
    model=model,
    train_dataloader=train_loader,
    debug=True,
)
```

## Version History

### v2.0.0 (Current)
- Complete rewrite with transformer architecture
- Physics-informed constraints
- Multi-modal fusion improvements
- Comprehensive evaluation suite

### v1.0.0
- Initial release with VAE-based architecture
- Basic video prediction capabilities
- Limited multimodal support

## Support

For questions and support:
- **Email**: nikjois@llamasearch.ai
- **GitHub Issues**: [GitHub Repository](https://github.com/nikjois/openworld-multimodal/issues)
- **Documentation**: [Full Documentation](https://openworld-multimodal.readthedocs.io) 