"""
OpenWorld-Multimodal Training Framework
Advanced trainer with distributed support, checkpointing, and comprehensive evaluation.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import os
import time
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    SummaryWriter = None
    HAS_TENSORBOARD = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    wandb = None
    HAS_WANDB = False

from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator

from ..models.transformer_world_model import TransformerWorldModel
from ..training.losses import MultimodalLoss
from ..utils.logging import get_logger

logger = get_logger(__name__)


class WorldModelTrainer:
    """
    Comprehensive trainer for OpenWorld-Multimodal system.
    
    Features:
    - Distributed training support
    - Mixed precision training
    - Gradient accumulation
    - Advanced scheduling
    - Comprehensive checkpointing
    - Real-time monitoring
    """
    
    def __init__(
        self,
        model: TransformerWorldModel,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        loss_fn: Optional[MultimodalLoss] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        accelerator: Optional[Accelerator] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.accelerator = accelerator or Accelerator()
        
        # Setup loss function
        self.loss_fn = loss_fn or MultimodalLoss()
        
        # Setup optimizer if not provided
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=config.get('learning_rate', 1e-4),
                weight_decay=config.get('weight_decay', 0.01),
                betas=config.get('betas', (0.9, 0.95)),
            )
        else:
            self.optimizer = optimizer
            
        # Setup scheduler if not provided
        if scheduler is None and config:
            total_steps = len(train_dataloader) * config.get('num_epochs', 100)
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=config.get('warmup_steps', total_steps // 10),
                num_training_steps=total_steps,
            )
        else:
            self.scheduler = scheduler
            
        # Training configuration
        self.config = config or {}
        self.gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        self.max_grad_norm = self.config.get('max_grad_norm', 1.0)
        self.eval_frequency = self.config.get('eval_frequency', 1000)
        self.save_frequency = self.config.get('save_frequency', 5000)
        self.log_frequency = self.config.get('log_frequency', 100)
        
        # Prepare with accelerator
        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.val_dataloader,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.val_dataloader,
            self.scheduler,
        )
        
        # Setup logging
        self.setup_logging()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        logger.info(f"Trainer initialized with {self.accelerator.num_processes} processes")
        
    def setup_logging(self):
        """Setup logging with TensorBoard and Weights & Biases."""
        if self.accelerator.is_main_process:
            # Setup output directory
            self.output_dir = Path(self.config.get('output_dir', './outputs'))
            self.output_dir.mkdir(exist_ok=True)
            
            # Setup TensorBoard
            if HAS_TENSORBOARD:
                self.writer = SummaryWriter(log_dir=self.output_dir / 'tensorboard')
            else:
                self.writer = None
                logger.warning("TensorBoard not available. Logging will be limited.")
            
            # Setup Weights & Biases
            if self.config.get('use_wandb', False) and HAS_WANDB:
                wandb.init(
                    project=self.config.get('wandb_project', 'openworld-multimodal'),
                    name=self.config.get('run_name', f'run_{int(time.time())}'),
                    config=self.config,
                )
                wandb.watch(self.accelerator.unwrap_model(self.model))
            elif self.config.get('use_wandb', False):
                logger.warning("Weights & Biases not available. Skipping wandb setup.")
                
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        
        for step, batch in enumerate(self.train_dataloader):
            with self.accelerator.accumulate(self.model):
                # Forward pass
                outputs = self.model(
                    video=batch['video'],
                    audio=batch['audio'],
                    timesteps=batch.get('timesteps'),
                )
                
                # Compute loss
                loss_dict = self.loss_fn(outputs, batch)
                loss = loss_dict['total_loss']
                
                # Backward pass
                self.accelerator.backward(loss)
                
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), 
                        self.max_grad_norm
                    )
                
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                
            # Logging
            if self.accelerator.sync_gradients:
                self.global_step += 1
                epoch_losses.append(loss.item())
                
                if self.global_step % self.log_frequency == 0:
                    self._log_metrics({
                        'train/loss': loss.item(),
                        'train/lr': self.optimizer.param_groups[0]['lr'],
                        'train/step': self.global_step,
                        **{f'train/{k}': v.item() for k, v in loss_dict.items() if k != 'total_loss'}
                    })
                    
                # Evaluation
                if self.global_step % self.eval_frequency == 0 and self.val_dataloader:
                    val_metrics = self.evaluate()
                    self._log_metrics(val_metrics)
                    
                    # Save best model
                    if val_metrics['val/loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['val/loss']
                        self.save_checkpoint('best_model')
                        
                # Save checkpoint
                if self.global_step % self.save_frequency == 0:
                    self.save_checkpoint(f'checkpoint_step_{self.global_step}')
                    
        return {'train_loss': sum(epoch_losses) / len(epoch_losses)}
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on validation set."""
        if not self.val_dataloader:
            return {}
            
        self.model.eval()
        val_losses = []
        
        for batch in self.val_dataloader:
            outputs = self.model(
                video=batch['video'],
                audio=batch['audio'],
                timesteps=batch.get('timesteps'),
            )
            
            loss_dict = self.loss_fn(outputs, batch)
            val_losses.append(loss_dict['total_loss'].item())
            
        avg_val_loss = sum(val_losses) / len(val_losses)
        
        self.model.train()
        return {'val/loss': avg_val_loss}
    
    def train(self, num_epochs: int):
        """Main training loop."""
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Train epoch
            epoch_metrics = self.train_epoch()
            
            # Log epoch metrics
            self._log_metrics({
                'epoch': epoch,
                **epoch_metrics
            })
            
            logger.info(f"Epoch {epoch + 1} completed. Train loss: {epoch_metrics['train_loss']:.4f}")
            
        # Final evaluation and save
        if self.val_dataloader:
            final_metrics = self.evaluate()
            self._log_metrics(final_metrics)
            
        self.save_checkpoint('final_model')
        logger.info("Training completed!")
        
    def save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint."""
        if not self.accelerator.is_main_process:
            return
            
        checkpoint_path = self.output_dir / f'{checkpoint_name}.pt'
        
        # Prepare checkpoint data
        checkpoint_data = {
            'model_state_dict': self.accelerator.unwrap_model(self.model).state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer and scheduler states
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        # Load training state
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to all configured loggers."""
        if not self.accelerator.is_main_process:
            return
            
        # TensorBoard logging
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, self.global_step)
            
        # Weights & Biases logging
        if self.config.get('use_wandb', False) and HAS_WANDB:
            wandb.log(metrics, step=self.global_step)
            
        # Console logging
        if self.global_step % self.log_frequency == 0:
            log_str = f"Step {self.global_step}: " + ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            logger.info(log_str)


def create_trainer(
    model: TransformerWorldModel,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
    config: Optional[Dict[str, Any]] = None,
) -> WorldModelTrainer:
    """
    Factory function to create a trainer with default configuration.
    
    Args:
        model: The world model to train
        train_dataloader: Training data loader
        val_dataloader: Optional validation data loader
        config: Training configuration dictionary
        
    Returns:
        Configured WorldModelTrainer instance
    """
    # Default configuration
    default_config = {
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'betas': (0.9, 0.95),
        'gradient_accumulation_steps': 1,
        'max_grad_norm': 1.0,
        'eval_frequency': 1000,
        'save_frequency': 5000,
        'log_frequency': 100,
        'num_epochs': 100,
        'warmup_steps': None,  # Will be computed automatically
        'output_dir': './outputs',
        'use_wandb': False,
        'wandb_project': 'openworld-multimodal',
    }
    
    if config:
        default_config.update(config)
        
    # Create accelerator for distributed training
    accelerator = Accelerator(
        gradient_accumulation_steps=default_config['gradient_accumulation_steps'],
        mixed_precision='fp16' if default_config.get('use_fp16', False) else None,
        log_with=['tensorboard'] + (['wandb'] if default_config['use_wandb'] else []),
    )
    
    return WorldModelTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        accelerator=accelerator,
        config=default_config,
    ) 