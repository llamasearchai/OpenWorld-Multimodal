"""
OpenWorld-Multimodal Evaluation CLI
Command line interface for evaluating the world model.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import click
import torch
import json
from pathlib import Path
from typing import Optional

from ..models.transformer_world_model import TransformerWorldModel
from ..evaluation.benchmarks import run_benchmark_suite
from ..data.hf_datasets import create_dataloaders
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
    '--dataset',
    type=str,
    default='synthetic',
    help='Dataset to evaluate on'
)
@click.option(
    '--output-dir',
    type=click.Path(),
    default='./evaluation_results',
    help='Output directory for evaluation results'
)
@click.option(
    '--batch-size',
    type=int,
    default=4,
    help='Batch size for evaluation'
)
@click.option(
    '--num-samples',
    type=int,
    default=None,
    help='Number of samples to evaluate (None for all)'
)
def evaluate_cmd(
    checkpoint: str,
    dataset: str,
    output_dir: str,
    batch_size: int,
    num_samples: Optional[int],
):
    """Evaluate the OpenWorld-Multimodal model."""
    try:
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
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
        
        # Create data loader
        click.echo("Creating evaluation data loader...")
        _, _, test_loader = create_dataloaders(
            dataset_name=dataset,
            batch_size=batch_size,
            num_workers=4,
            max_samples=num_samples,
        )
        
        # Run evaluation
        click.echo("🔍 Starting evaluation...")
        results = run_benchmark_suite(
            model=model,
            test_dataloader=test_loader,
            device=device,
            output_dir=Path(output_dir),
        )
        
        # Print results summary
        click.echo("\n📈 Evaluation Results:")
        click.echo("=" * 50)
        
        for benchmark_name, result in results.items():
            click.echo(f"\n{benchmark_name.replace('_', ' ').title()}:")
            for metric_name, value in result.metrics.items():
                if isinstance(value, float):
                    click.echo(f"  {metric_name}: {value:.4f}")
                else:
                    click.echo(f"  {metric_name}: {value}")
        
        click.echo(f"\n✅ Evaluation completed! Results saved to {output_dir}")
        
    except Exception as e:
        click.echo(f"❌ Evaluation failed: {e}")
        logger.exception("Evaluation failed")
        raise click.ClickException(str(e)) 