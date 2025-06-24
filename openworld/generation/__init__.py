"""
OpenWorld-Multimodal Generation Module

Advanced sampling and generation strategies for multimodal world modeling.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

from .sampler import (
    WorldModelSampler,
    SamplingConfig,
)

from .beam_search import (
    BeamSearch,
    BeamSearchHypothesis,
    MultimodalBeamSearch,
)

# Create alias for backwards compatibility
BeamSearchGenerator = BeamSearch

__all__ = [
    'WorldModelSampler',
    'SamplingConfig',
    'BeamSearch',
    'BeamSearchHypothesis',
    'MultimodalBeamSearch',
    'BeamSearchGenerator',
] 