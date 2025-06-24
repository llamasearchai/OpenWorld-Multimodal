"""
OpenWorld-Multimodal Benchmarking Suite
Comprehensive evaluation benchmarks for multimodal world modeling.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import time
from dataclasses import dataclass
from tqdm import tqdm

from ..models.transformer_world_model import TransformerWorldModel
from .perceptual_metrics import PerceptualMetrics
from .physics_metrics import PhysicsMetrics
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    metrics: Dict[str, float]
    execution_time: float
    num_samples: int
    metadata: Dict[str, Any]


class WorldModelBenchmark:
    """
    Comprehensive benchmark suite for OpenWorld-Multimodal system.
    
    Evaluates:
    - Video prediction quality (PSNR, SSIM, LPIPS)
    - Audio generation quality (spectral metrics)
    - Physics consistency
    - Temporal coherence
    - Cross-modal alignment
    - Computational efficiency
    """
    
    def __init__(
        self,
        model: TransformerWorldModel,
        device: torch.device,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.config = config or {}
        
        # Initialize metrics
        self.perceptual_metrics = PerceptualMetrics(device=device)
        self.physics_metrics = PhysicsMetrics(device=device)
        
        # Benchmark configuration
        self.num_prediction_steps = self.config.get('num_prediction_steps', 10)
        self.batch_size = self.config.get('batch_size', 4)
        self.resolution = self.config.get('resolution', 256)
        
        logger.info("WorldModelBenchmark initialized")
        
    def run_full_benchmark(
        self,
        test_dataloader,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, BenchmarkResult]:
        """
        Run complete benchmark suite.
        
        Args:
            test_dataloader: Test data loader
            output_dir: Optional directory to save results
            
        Returns:
            Dictionary of benchmark results
        """
        results = {}
        
        # Video prediction benchmark
        logger.info("Running video prediction benchmark...")
        results['video_prediction'] = self.benchmark_video_prediction(test_dataloader)
        
        # Audio generation benchmark
        logger.info("Running audio generation benchmark...")
        results['audio_generation'] = self.benchmark_audio_generation(test_dataloader)
        
        # Physics consistency benchmark
        logger.info("Running physics consistency benchmark...")
        results['physics_consistency'] = self.benchmark_physics_consistency(test_dataloader)
        
        # Temporal coherence benchmark
        logger.info("Running temporal coherence benchmark...")
        results['temporal_coherence'] = self.benchmark_temporal_coherence(test_dataloader)
        
        # Cross-modal alignment benchmark
        logger.info("Running cross-modal alignment benchmark...")
        results['cross_modal_alignment'] = self.benchmark_cross_modal_alignment(test_dataloader)
        
        # Computational efficiency benchmark
        logger.info("Running efficiency benchmark...")
        results['computational_efficiency'] = self.benchmark_computational_efficiency(test_dataloader)
        
        # Save results if output directory specified
        if output_dir:
            self.save_benchmark_results(results, output_dir)
            
        return results
    
    @torch.no_grad()
    def benchmark_video_prediction(self, dataloader) -> BenchmarkResult:
        """Benchmark video prediction quality."""
        self.model.eval()
        
        start_time = time.time()
        all_metrics = []
        num_samples = 0
        
        for batch in tqdm(dataloader, desc="Video Prediction"):
            video = batch['video'].to(self.device)
            audio = batch['audio'].to(self.device)
            
            # Split into context and target
            context_len = video.size(1) // 2
            context_video = video[:, :context_len]
            context_audio = audio[:, :context_len]
            target_video = video[:, context_len:]
            
            # Generate predictions
            outputs = self.model(
                video=context_video,
                audio=context_audio,
                future_steps=self.num_prediction_steps,
            )
            
            if 'future_video' in outputs:
                pred_video = outputs['future_video']
                
                # Compute perceptual metrics
                batch_metrics = self.perceptual_metrics.compute_video_metrics(
                    pred_video, target_video[:, :pred_video.size(1)]
                )
                all_metrics.append(batch_metrics)
                num_samples += video.size(0)
                
        # Aggregate metrics
        aggregated_metrics = self._aggregate_metrics(all_metrics)
        execution_time = time.time() - start_time
        
        return BenchmarkResult(
            name="video_prediction",
            metrics=aggregated_metrics,
            execution_time=execution_time,
            num_samples=num_samples,
            metadata={
                'prediction_steps': self.num_prediction_steps,
                'resolution': self.resolution,
            }
        )
    
    @torch.no_grad()
    def benchmark_audio_generation(self, dataloader) -> BenchmarkResult:
        """Benchmark audio generation quality."""
        self.model.eval()
        
        start_time = time.time()
        all_metrics = []
        num_samples = 0
        
        for batch in tqdm(dataloader, desc="Audio Generation"):
            video = batch['video'].to(self.device)
            audio = batch['audio'].to(self.device)
            
            # Split into context and target
            context_len = audio.size(1) // 2
            context_video = video[:, :context_len]
            context_audio = audio[:, :context_len]
            target_audio = audio[:, context_len:]
            
            # Generate predictions
            outputs = self.model(
                video=context_video,
                audio=context_audio,
                future_steps=self.num_prediction_steps,
            )
            
            if 'future_audio' in outputs:
                pred_audio = outputs['future_audio']
                
                # Compute audio metrics
                batch_metrics = self.perceptual_metrics.compute_audio_metrics(
                    pred_audio, target_audio[:, :pred_audio.size(1)]
                )
                all_metrics.append(batch_metrics)
                num_samples += audio.size(0)
                
        # Aggregate metrics
        aggregated_metrics = self._aggregate_metrics(all_metrics)
        execution_time = time.time() - start_time
        
        return BenchmarkResult(
            name="audio_generation",
            metrics=aggregated_metrics,
            execution_time=execution_time,
            num_samples=num_samples,
            metadata={
                'prediction_steps': self.num_prediction_steps,
            }
        )
    
    @torch.no_grad()
    def benchmark_physics_consistency(self, dataloader) -> BenchmarkResult:
        """Benchmark physics consistency of predictions."""
        self.model.eval()
        
        start_time = time.time()
        all_metrics = []
        num_samples = 0
        
        for batch in tqdm(dataloader, desc="Physics Consistency"):
            video = batch['video'].to(self.device)
            audio = batch['audio'].to(self.device)
            
            # Generate predictions
            outputs = self.model(
                video=video,
                audio=audio,
                future_steps=self.num_prediction_steps,
                return_intermediates=True,
            )
            
            if 'future_video' in outputs:
                # Compute physics metrics
                batch_metrics = self.physics_metrics.compute_physics_consistency(
                    outputs['future_video'], 
                    outputs.get('latent'),
                )
                all_metrics.append(batch_metrics)
                num_samples += video.size(0)
                
        # Aggregate metrics
        aggregated_metrics = self._aggregate_metrics(all_metrics)
        execution_time = time.time() - start_time
        
        return BenchmarkResult(
            name="physics_consistency",
            metrics=aggregated_metrics,
            execution_time=execution_time,
            num_samples=num_samples,
            metadata={
                'prediction_steps': self.num_prediction_steps,
            }
        )
    
    @torch.no_grad()
    def benchmark_temporal_coherence(self, dataloader) -> BenchmarkResult:
        """Benchmark temporal coherence of video predictions."""
        self.model.eval()
        
        start_time = time.time()
        all_metrics = []
        num_samples = 0
        
        for batch in tqdm(dataloader, desc="Temporal Coherence"):
            video = batch['video'].to(self.device)
            audio = batch['audio'].to(self.device)
            
            # Generate predictions
            outputs = self.model(
                video=video,
                audio=audio,
                future_steps=self.num_prediction_steps,
            )
            
            if 'future_video' in outputs:
                pred_video = outputs['future_video']
                
                # Compute temporal coherence metrics
                batch_metrics = self._compute_temporal_coherence(pred_video)
                all_metrics.append(batch_metrics)
                num_samples += video.size(0)
                
        # Aggregate metrics
        aggregated_metrics = self._aggregate_metrics(all_metrics)
        execution_time = time.time() - start_time
        
        return BenchmarkResult(
            name="temporal_coherence",
            metrics=aggregated_metrics,
            execution_time=execution_time,
            num_samples=num_samples,
            metadata={
                'prediction_steps': self.num_prediction_steps,
            }
        )
    
    @torch.no_grad()
    def benchmark_cross_modal_alignment(self, dataloader) -> BenchmarkResult:
        """Benchmark cross-modal alignment between video and audio."""
        self.model.eval()
        
        start_time = time.time()
        all_metrics = []
        num_samples = 0
        
        for batch in tqdm(dataloader, desc="Cross-Modal Alignment"):
            video = batch['video'].to(self.device)
            audio = batch['audio'].to(self.device)
            
            # Generate predictions
            outputs = self.model(
                video=video,
                audio=audio,
                future_steps=self.num_prediction_steps,
                return_intermediates=True,
            )
            
            if 'future_video' in outputs and 'future_audio' in outputs:
                # Compute cross-modal alignment metrics
                batch_metrics = self._compute_cross_modal_alignment(
                    outputs['future_video'],
                    outputs['future_audio'],
                    outputs.get('intermediates', [])
                )
                all_metrics.append(batch_metrics)
                num_samples += video.size(0)
                
        # Aggregate metrics
        aggregated_metrics = self._aggregate_metrics(all_metrics)
        execution_time = time.time() - start_time
        
        return BenchmarkResult(
            name="cross_modal_alignment",
            metrics=aggregated_metrics,
            execution_time=execution_time,
            num_samples=num_samples,
            metadata={
                'prediction_steps': self.num_prediction_steps,
            }
        )
    
    def benchmark_computational_efficiency(self, dataloader) -> BenchmarkResult:
        """Benchmark computational efficiency and memory usage."""
        self.model.eval()
        
        start_time = time.time()
        forward_times = []
        memory_usage = []
        num_samples = 0
        
        # Warmup
        for i, batch in enumerate(dataloader):
            if i >= 3:  # Only warmup for 3 batches
                break
            video = batch['video'].to(self.device)
            audio = batch['audio'].to(self.device)
            _ = self.model(video=video, audio=audio)
            
        # Actual benchmarking
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Efficiency"):
                video = batch['video'].to(self.device)
                audio = batch['audio'].to(self.device)
                
                # Measure forward pass time
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                forward_start = time.time()
                
                outputs = self.model(
                    video=video,
                    audio=audio,
                    future_steps=self.num_prediction_steps,
                )
                
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                forward_time = time.time() - forward_start
                forward_times.append(forward_time)
                
                # Measure memory usage
                if self.device.type == 'cuda':
                    memory_usage.append(torch.cuda.max_memory_allocated() / 1024**3)  # GB
                    
                num_samples += video.size(0)
                
        # Compute efficiency metrics
        metrics = {
            'avg_forward_time': np.mean(forward_times),
            'std_forward_time': np.std(forward_times),
            'throughput_samples_per_sec': num_samples / sum(forward_times),
            'throughput_fps': (num_samples * self.num_prediction_steps) / sum(forward_times),
        }
        
        if memory_usage:
            metrics.update({
                'avg_memory_gb': np.mean(memory_usage),
                'max_memory_gb': np.max(memory_usage),
            })
            
        execution_time = time.time() - start_time
        
        return BenchmarkResult(
            name="computational_efficiency",
            metrics=metrics,
            execution_time=execution_time,
            num_samples=num_samples,
            metadata={
                'device': str(self.device),
                'batch_size': self.batch_size,
            }
        )
    
    def _aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across batches."""
        if not metrics_list:
            return {}
            
        aggregated = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if key in m]
            if values:
                aggregated[f'{key}_mean'] = np.mean(values)
                aggregated[f'{key}_std'] = np.std(values)
                aggregated[f'{key}_min'] = np.min(values)
                aggregated[f'{key}_max'] = np.max(values)
                
        return aggregated
    
    def _compute_temporal_coherence(self, video: torch.Tensor) -> Dict[str, float]:
        """Compute temporal coherence metrics."""
        B, T, C, H, W = video.shape
        
        # Compute frame-to-frame differences
        frame_diffs = torch.abs(video[:, 1:] - video[:, :-1])
        
        # Temporal smoothness (lower is better)
        temporal_smoothness = frame_diffs.mean().item()
        
        # Temporal variance (consistency measure)
        temporal_variance = frame_diffs.var().item()
        
        return {
            'temporal_smoothness': temporal_smoothness,
            'temporal_variance': temporal_variance,
        }
    
    def _compute_cross_modal_alignment(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        intermediates: List[torch.Tensor],
    ) -> Dict[str, float]:
        """Compute cross-modal alignment metrics."""
        # Simplified cross-modal alignment computation
        # In practice, this would involve more sophisticated metrics
        
        # Compute correlation between video and audio features
        video_features = video.mean(dim=(3, 4))  # Average spatial dimensions
        audio_features = audio
        
        # Temporal correlation
        correlation = torch.corrcoef(
            torch.cat([
                video_features.flatten(1),
                audio_features.flatten(1)
            ], dim=1)
        )
        
        # Extract cross-modal correlation (off-diagonal blocks)
        cross_modal_corr = correlation[:video_features.size(1), video_features.size(1):].abs().mean().item()
        
        return {
            'cross_modal_correlation': cross_modal_corr,
        }
    
    def save_benchmark_results(
        self,
        results: Dict[str, BenchmarkResult],
        output_dir: Path,
    ):
        """Save benchmark results to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual results
        for name, result in results.items():
            result_dict = {
                'name': result.name,
                'metrics': result.metrics,
                'execution_time': result.execution_time,
                'num_samples': result.num_samples,
                'metadata': result.metadata,
            }
            
            with open(output_dir / f'{name}_results.json', 'w') as f:
                json.dump(result_dict, f, indent=2)
                
        # Save summary
        summary = {
            'benchmark_results': {
                name: result.metrics for name, result in results.items()
            },
            'total_execution_time': sum(r.execution_time for r in results.values()),
            'total_samples': sum(r.num_samples for r in results.values()),
        }
        
        with open(output_dir / 'benchmark_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Benchmark results saved to {output_dir}")


def run_benchmark_suite(
    model: TransformerWorldModel,
    test_dataloader,
    device: torch.device,
    output_dir: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, BenchmarkResult]:
    """
    Convenience function to run complete benchmark suite.
    
    Args:
        model: The world model to benchmark
        test_dataloader: Test data loader
        device: Device to run benchmarks on
        output_dir: Optional directory to save results
        config: Benchmark configuration
        
    Returns:
        Dictionary of benchmark results
    """
    benchmark = WorldModelBenchmark(model, device, config)
    return benchmark.run_full_benchmark(test_dataloader, output_dir) 