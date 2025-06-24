"""
OpenWorld-Multimodal Command Line Interface

Author: Nik Jois <nikjois@llamasearch.ai>
"""

from .main import main
from .train import train_cmd
from .evaluate import evaluate_cmd
from .generate import generate_cmd

__all__ = [
    'main',
    'train_cmd',
    'evaluate_cmd', 
    'generate_cmd',
] 