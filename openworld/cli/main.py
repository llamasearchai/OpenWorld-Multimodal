"""
OpenWorld-Multimodal Main CLI Entry Point
Command line interface for training, evaluation, and generation.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import click
import sys
from pathlib import Path
import logging

from ..utils.logging import setup_logging

logger = logging.getLogger(__name__)


@click.group()
@click.option(
    '--verbose', '-v',
    count=True,
    help='Increase verbosity (use -v, -vv, or -vvv)'
)
@click.option(
    '--log-file',
    type=click.Path(),
    help='Log file path'
)
@click.version_option(version='2.0.0', prog_name='OpenWorld-Multimodal')
def cli(verbose: int, log_file: str):
    """
    OpenWorld-Multimodal: Advanced Multimodal World Modeling System
    
    A state-of-the-art transformer-based system for multimodal video and audio
    prediction with physics-informed constraints.
    
    Author: Nik Jois <nikjois@llamasearch.ai>
    """
    # Setup logging based on verbosity
    log_level = logging.WARNING
    if verbose == 1:
        log_level = logging.INFO
    elif verbose >= 2:
        log_level = logging.DEBUG
        
    setup_logging(level=log_level, log_file=log_file)
    
    logger.info("OpenWorld-Multimodal CLI initialized")


# Conditionally add subcommands if dependencies are available
try:
    from .train import train_cmd
    cli.add_command(train_cmd, name='train')
except ImportError as e:
    logger.warning(f"Training command not available due to missing dependencies: {e}")

try:
    from .evaluate import evaluate_cmd
    cli.add_command(evaluate_cmd, name='evaluate')
except ImportError as e:
    logger.warning(f"Evaluation command not available due to missing dependencies: {e}")

try:
    from .generate import generate_cmd
    cli.add_command(generate_cmd, name='generate')
except ImportError as e:
    logger.warning(f"Generation command not available due to missing dependencies: {e}")


@cli.command()
@click.option(
    '--config-path',
    type=click.Path(exists=True),
    help='Path to configuration file'
)
def info(config_path: str):
    """Display system information and configuration."""
    import torch
    import platform
    
    click.echo("=== OpenWorld-Multimodal System Information ===")
    click.echo(f"Version: 2.0.0")
    click.echo(f"Author: Nik Jois <nikjois@llamasearch.ai>")
    click.echo()
    
    # System information
    click.echo("=== System Information ===")
    click.echo(f"Platform: {platform.platform()}")
    click.echo(f"Python: {platform.python_version()}")
    click.echo(f"PyTorch: {torch.__version__}")
    click.echo(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        click.echo(f"CUDA Version: {torch.version.cuda}")
        click.echo(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            click.echo(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    click.echo()
    
    # Model information
    click.echo("=== Model Information ===")
    try:
        from ..models.transformer_world_model import TransformerWorldModel
        model = TransformerWorldModel()
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        click.echo(f"Total Parameters: {total_params:,}")
        click.echo(f"Trainable Parameters: {trainable_params:,}")
        click.echo(f"Model Size: {total_params * 4 / 1024 / 1024:.1f} MB (FP32)")
        click.echo()
    except Exception as e:
        click.echo(f"Could not load model: {e}")
    
    # Configuration information
    if config_path:
        click.echo("=== Configuration ===")
        try:
            import json
            with open(config_path) as f:
                config = json.load(f)
            for key, value in config.items():
                click.echo(f"{key}: {value}")
        except Exception as e:
            click.echo(f"Could not load config: {e}")


@cli.command()
def demo():
    """Run a quick demo of the OpenWorld-Multimodal system."""
    import torch
    import numpy as np
    
    try:
        from ..models.transformer_world_model import TransformerWorldModel
        
        click.echo("Starting OpenWorld-Multimodal Demo...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        click.echo(f"Using device: {device}")
        
        # Create model
        click.echo("Creating model...")
        model = TransformerWorldModel(
            img_size=128,  # Smaller for demo
            embed_dim=512,
            depth=6,
            num_heads=8,
        ).to(device)
        
        # Create dummy data
        click.echo("Creating dummy data...")
        batch_size = 2
        seq_len = 8
        video = torch.randn(batch_size, seq_len, 3, 128, 128).to(device)
        audio = torch.randn(batch_size, seq_len, 128).to(device)
        
        # Forward pass
        click.echo("Running forward pass...")
        model.eval()
        with torch.no_grad():
            outputs = model(
                video=video,
                audio=audio,
                future_steps=4,
            )
        
        click.echo("✅ Demo completed successfully!")
        click.echo(f"Input video shape: {video.shape}")
        click.echo(f"Input audio shape: {audio.shape}")
        
        if 'future_video' in outputs:
            click.echo(f"Generated video shape: {outputs['future_video'].shape}")
        if 'future_audio' in outputs:
            click.echo(f"Generated audio shape: {outputs['future_audio'].shape}")
            
        click.echo("🎉 OpenWorld-Multimodal is working correctly!")
        
    except Exception as e:
        click.echo(f"❌ Demo failed: {e}")
        if hasattr(e, '__traceback__'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        logger.exception("Unhandled exception in CLI")
        sys.exit(1)


if __name__ == '__main__':
    main() 