"""
OpenWorld-Multimodal Training CLI
Command line interface for training the world model.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import click
import torch
import json
from pathlib import Path
from typing import Optional, Dict, Any

from ..models.transformer_world_model import TransformerWorldModel
from ..data.hf_datasets import create_dataloaders
from ..training.trainer import create_trainer
from ..utils.logging import get_logger

logger = get_logger(__name__)


@click.command()
@click.option(
    '--config',
    type=click.Path(exists=True),
    required=True,
    help='Path to training configuration file'
)
@click.option(
    '--output-dir',
    type=click.Path(),
    default='./outputs',
    help='Output directory for checkpoints and logs'
)
@click.option(
    '--resume-from',
    type=click.Path(exists=True),
    help='Resume training from checkpoint'
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Perform a dry run without actual training'
)
@click.option(
    '--debug',
    is_flag=True,
    help='Enable debug mode with reduced data'
)
def train_cmd(
    config: str,
    output_dir: str,
    resume_from: Optional[str],
    dry_run: bool,
    debug: bool,
):
    """Train the OpenWorld-Multimodal model."""
    try:
        # Load configuration
        config_path = Path(config)
        with open(config_path) as f:
            config_dict = json.load(f)
            
        logger.info(f"Loaded configuration from {config_path}")
        
        # Override config with CLI options
        config_dict['output_dir'] = output_dir
        if debug:
            config_dict['debug'] = True
            config_dict['max_samples'] = 100
            config_dict['num_epochs'] = 2
            
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        if dry_run:
            click.echo("🔍 Dry run mode - validating configuration...")
            
        # Create model
        click.echo("🏗️  Creating model...")
        model_config = config_dict.get('model', {})
        model = TransformerWorldModel(**model_config)
        
        logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Create data loaders
        click.echo("Creating data loaders...")
        train_loader, val_loader, test_loader = create_dataloaders(
            dataset_name=config_dict.get('dataset', 'synthetic'),
            batch_size=config_dict.get('batch_size', 4),
            num_workers=config_dict.get('num_workers', 4),
            max_samples=config_dict.get('max_samples', None),
        )
        
        logger.info(f"Created data loaders - Train: {len(train_loader)}, Val: {len(val_loader)}")
        
        if dry_run:
            click.echo("✅ Dry run completed successfully!")
            click.echo(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            click.echo(f"Training batches: {len(train_loader)}")
            click.echo(f"Validation batches: {len(val_loader)}")
            return
            
        # Create trainer
        click.echo("Creating trainer...")
        trainer = create_trainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            config=config_dict,
        )
        
        # Resume from checkpoint if specified
        if resume_from:
            click.echo(f"📁 Resuming from checkpoint: {resume_from}")
            trainer.load_checkpoint(resume_from)
            
        # Start training
        num_epochs = config_dict.get('num_epochs', 100)
        click.echo(f"🎯 Starting training for {num_epochs} epochs...")
        
        trainer.train(num_epochs)
        
        click.echo("🎉 Training completed successfully!")
        
    except Exception as e:
        click.echo(f"❌ Training failed: {e}")
        logger.exception("Training failed")
        raise click.ClickException(str(e))


def validate_config(config_dict: Dict[str, Any]) -> bool:
    """Validate training configuration."""
    required_keys = ['model', 'batch_size', 'num_epochs']
    
    for key in required_keys:
        if key not in config_dict:
            raise ValueError(f"Missing required config key: {key}")
            
    return True 