"""
OpenWorld-Multimodal Generation CLI
Command line interface for generating videos and audio.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import click
import torch
import json
import numpy as np
from pathlib import Path
from typing import Optional

from ..models.transformer_world_model import TransformerWorldModel
from ..generation.sampler import create_sampler, SamplingConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


@click.command()
@click.option(
    '--checkpoint',
    type=click.Path(exists=True),
    required=True,
    help='Path to model checkpoint'
)
@click.option(
    '--output-dir',
    type=click.Path(),
    default='./generated',
    help='Output directory for generated content'
)
@click.option(
    '--num-steps',
    type=int,
    default=10,
    help='Number of future steps to generate'
)
@click.option(
    '--num-samples',
    type=int,
    default=1,
    help='Number of samples to generate'
)
@click.option(
    '--temperature',
    type=float,
    default=1.0,
    help='Sampling temperature'
)
@click.option(
    '--top-k',
    type=int,
    default=None,
    help='Top-k sampling parameter'
)
@click.option(
    '--top-p',
    type=float,
    default=None,
    help='Top-p (nucleus) sampling parameter'
)
@click.option(
    '--seed',
    type=int,
    default=None,
    help='Random seed for reproducibility'
)
def generate_cmd(
    checkpoint: str,
    output_dir: str,
    num_steps: int,
    num_samples: int,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    seed: Optional[int],
):
    """Generate videos and audio with the OpenWorld-Multimodal model."""
    try:
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            click.echo(f"🎯 Using seed: {seed}")
        
        # Load checkpoint
        click.echo(f"📁 Loading checkpoint: {checkpoint}")
        checkpoint_data = torch.load(checkpoint, map_location=device)
        
        # Create model
        click.echo("🏗️  Creating model...")
        model_config = checkpoint_data.get('config', {}).get('model', {})
        model = TransformerWorldModel(**model_config)
        model.load_state_dict(checkpoint_data['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Create sampler
        click.echo("🎨 Creating sampler...")
        sampling_config = SamplingConfig(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_samples=num_samples,
            seed=seed,
        )
        sampler = create_sampler(model, device, sampling_config.__dict__)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate context (dummy data for demo)
        click.echo("Creating context data...")
        batch_size = 1
        context_length = 8
        img_size = model_config.get('img_size', 256)
        audio_dim = model_config.get('audio_dim', 128)
        
        context_video = torch.randn(batch_size, context_length, 3, img_size, img_size).to(device)
        context_audio = torch.randn(batch_size, context_length, audio_dim).to(device)
        
        # Generate samples
        click.echo(f"Generating {num_samples} samples with {num_steps} future steps...")
        
        for sample_idx in range(num_samples):
            click.echo(f"  Generating sample {sample_idx + 1}/{num_samples}...")
            
            generated = sampler.generate(
                context_video=context_video,
                context_audio=context_audio,
                num_steps=num_steps,
            )
            
            # Save generated content
            sample_dir = output_path / f'sample_{sample_idx:03d}'
            sample_dir.mkdir(exist_ok=True)
            
            # Save video
            if 'generated_video' in generated:
                video_path = sample_dir / 'video.pt'
                torch.save(generated['generated_video'], video_path)
                click.echo(f"    💾 Video saved: {video_path}")
            
            # Save audio
            if 'generated_audio' in generated:
                audio_path = sample_dir / 'audio.pt'
                torch.save(generated['generated_audio'], audio_path)
                click.echo(f"    💾 Audio saved: {audio_path}")
            
            # Save metadata
            metadata = {
                'num_steps': num_steps,
                'temperature': temperature,
                'top_k': top_k,
                'top_p': top_p,
                'seed': seed,
                'shapes': {
                    'context_video': list(context_video.shape),
                    'context_audio': list(context_audio.shape),
                    'generated_video': list(generated['generated_video'].shape) if 'generated_video' in generated else None,
                    'generated_audio': list(generated['generated_audio'].shape) if 'generated_audio' in generated else None,
                }
            }
            
            metadata_path = sample_dir / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        click.echo(f"\n✅ Generation completed! {num_samples} samples saved to {output_dir}")
        click.echo(f"📁 Each sample contains video.pt, audio.pt, and metadata.json")
        
    except Exception as e:
        click.echo(f"❌ Generation failed: {e}")
        logger.exception("Generation failed")
        raise click.ClickException(str(e)) 