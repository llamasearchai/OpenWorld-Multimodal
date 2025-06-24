"""
Advanced Beam Search Implementation for OpenWorld-Multimodal

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any
import math
from dataclasses import dataclass


@dataclass
class BeamSearchHypothesis:
    """Single hypothesis in beam search."""
    sequence: torch.Tensor
    score: float
    log_probs: List[float]
    attention_weights: Optional[torch.Tensor] = None
    

class BeamSearch:
    """Advanced beam search for multimodal sequence generation."""
    
    def __init__(
        self,
        beam_size: int = 5,
        max_length: int = 100,
        length_penalty: float = 1.0,
        coverage_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        early_stopping: bool = True,
        min_length: int = 0,
        no_repeat_ngram_size: int = 0,
    ):
        self.beam_size = beam_size
        self.max_length = max_length
        self.length_penalty = length_penalty
        self.coverage_penalty = coverage_penalty
        self.repetition_penalty = repetition_penalty
        self.early_stopping = early_stopping
        self.min_length = min_length
        self.no_repeat_ngram_size = no_repeat_ngram_size
    
    def search(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **model_kwargs
    ) -> List[BeamSearchHypothesis]:
        """
        Perform beam search generation.
        
        Args:
            model: The generation model
            input_ids: Input token ids [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            **model_kwargs: Additional model arguments
            
        Returns:
            List of hypotheses for each batch item
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize beams
        beams = []
        for batch_idx in range(batch_size):
            beam_hypotheses = []
            for _ in range(self.beam_size):
                hypothesis = BeamSearchHypothesis(
                    sequence=input_ids[batch_idx:batch_idx+1].clone(),
                    score=0.0,
                    log_probs=[]
                )
                beam_hypotheses.append(hypothesis)
            beams.append(beam_hypotheses)
        
        # Generate sequences
        for step in range(self.max_length):
            all_candidates = []
            
            for batch_idx in range(batch_size):
                candidates = []
                
                for beam_idx, hypothesis in enumerate(beams[batch_idx]):
                    if self._is_finished(hypothesis):
                        candidates.append((hypothesis, 0.0))
                        continue
                    
                    # Get model predictions
                    with torch.no_grad():
                        outputs = model(
                            input_ids=hypothesis.sequence,
                            attention_mask=attention_mask[batch_idx:batch_idx+1] if attention_mask is not None else None,
                            **model_kwargs
                        )
                    
                    logits = outputs.logits[:, -1, :]  # Get last token logits
                    log_probs = F.log_softmax(logits, dim=-1)
                    
                    # Apply repetition penalty
                    if self.repetition_penalty != 1.0:
                        log_probs = self._apply_repetition_penalty(
                            log_probs, hypothesis.sequence
                        )
                    
                    # Get top-k candidates
                    top_log_probs, top_indices = torch.topk(
                        log_probs, k=self.beam_size, dim=-1
                    )
                    
                    for i in range(self.beam_size):
                        token_id = top_indices[0, i]
                        token_log_prob = top_log_probs[0, i].item()
                        
                        # Skip if would create n-gram repetition
                        if self._would_repeat_ngram(hypothesis.sequence, token_id):
                            continue
                        
                        # Create new hypothesis
                        new_sequence = torch.cat([
                            hypothesis.sequence,
                            token_id.unsqueeze(0).unsqueeze(0)
                        ], dim=1)
                        
                        new_score = hypothesis.score + token_log_prob
                        new_log_probs = hypothesis.log_probs + [token_log_prob]
                        
                        new_hypothesis = BeamSearchHypothesis(
                            sequence=new_sequence,
                            score=new_score,
                            log_probs=new_log_probs
                        )
                        
                        candidates.append((new_hypothesis, new_score))
                
                all_candidates.append(candidates)
            
            # Select top beams for each batch
            new_beams = []
            for batch_idx in range(batch_size):
                # Sort candidates by score
                candidates = all_candidates[batch_idx]
                candidates.sort(key=lambda x: x[1], reverse=True)
                
                # Select top beam_size candidates
                new_beam = []
                for hypothesis, score in candidates[:self.beam_size]:
                    new_beam.append(hypothesis)
                
                new_beams.append(new_beam)
            
            beams = new_beams
            
            # Check early stopping
            if self.early_stopping and self._all_finished(beams):
                break
        
        # Return best hypotheses
        results = []
        for batch_idx in range(batch_size):
            # Apply length penalty and select best
            best_hypothesis = max(
                beams[batch_idx],
                key=lambda h: self._compute_final_score(h)
            )
            results.append(best_hypothesis)
        
        return results
    
    def _is_finished(self, hypothesis: BeamSearchHypothesis) -> bool:
        """Check if hypothesis is finished (reached EOS or max length)."""
        return (
            hypothesis.sequence.shape[1] >= self.max_length or
            hypothesis.sequence[0, -1].item() == 2  # Assuming EOS token is 2
        )
    
    def _all_finished(self, beams: List[List[BeamSearchHypothesis]]) -> bool:
        """Check if all beams are finished."""
        for beam in beams:
            for hypothesis in beam:
                if not self._is_finished(hypothesis):
                    return False
        return True
    
    def _apply_repetition_penalty(
        self, log_probs: torch.Tensor, sequence: torch.Tensor
    ) -> torch.Tensor:
        """Apply repetition penalty to log probabilities."""
        if self.repetition_penalty == 1.0:
            return log_probs
        
        for token_id in sequence[0]:
            log_probs[0, token_id] /= self.repetition_penalty
        
        return log_probs
    
    def _would_repeat_ngram(
        self, sequence: torch.Tensor, new_token: torch.Tensor
    ) -> bool:
        """Check if adding new token would create forbidden n-gram repetition."""
        if self.no_repeat_ngram_size == 0:
            return False
        
        seq_len = sequence.shape[1]
        if seq_len < self.no_repeat_ngram_size - 1:
            return False
        
        # Check if the new n-gram would be a repetition
        ngram_start = seq_len - self.no_repeat_ngram_size + 1
        current_ngram = sequence[0, ngram_start:].tolist() + [new_token.item()]
        
        # Check all previous n-grams
        for i in range(seq_len - self.no_repeat_ngram_size + 1):
            prev_ngram = sequence[0, i:i + self.no_repeat_ngram_size].tolist()
            if prev_ngram == current_ngram:
                return True
        
        return False
    
    def _compute_final_score(self, hypothesis: BeamSearchHypothesis) -> float:
        """Compute final score with length penalty."""
        length = len(hypothesis.log_probs)
        if length == 0:
            return hypothesis.score
        
        # Length penalty
        length_penalty = ((5 + length) / 6) ** self.length_penalty
        
        return hypothesis.score / length_penalty


class MultimodalBeamSearch(BeamSearch):
    """Beam search specialized for multimodal generation."""
    
    def __init__(
        self,
        modality_weights: Dict[str, float] = None,
        cross_modal_penalty: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.modality_weights = modality_weights or {'video': 1.0, 'audio': 1.0}
        self.cross_modal_penalty = cross_modal_penalty
    
    def multimodal_search(
        self,
        model: torch.nn.Module,
        video: torch.Tensor,
        audio: torch.Tensor,
        **kwargs
    ) -> Dict[str, List[BeamSearchHypothesis]]:
        """
        Perform multimodal beam search.
        
        Args:
            model: Multimodal generation model
            video: Input video [batch_size, seq_len, ...]
            audio: Input audio [batch_size, seq_len, ...]
            
        Returns:
            Dictionary of hypotheses for each modality
        """
        batch_size = video.shape[0]
        device = video.device
        
        # Initialize multimodal beams
        video_beams = []
        audio_beams = []
        
        for batch_idx in range(batch_size):
            video_beam = []
            audio_beam = []
            
            for beam_idx in range(self.beam_size):
                # Initialize with input sequences
                video_hypothesis = BeamSearchHypothesis(
                    sequence=video[batch_idx:batch_idx+1].clone(),
                    score=0.0,
                    log_probs=[]
                )
                audio_hypothesis = BeamSearchHypothesis(
                    sequence=audio[batch_idx:batch_idx+1].clone(),
                    score=0.0,
                    log_probs=[]
                )
                
                video_beam.append(video_hypothesis)
                audio_beam.append(audio_hypothesis)
            
            video_beams.append(video_beam)
            audio_beams.append(audio_beam)
        
        # Generate sequences with cross-modal consistency
        for step in range(self.max_length):
            # Update beams with cross-modal information
            video_beams, audio_beams = self._multimodal_step(
                model, video_beams, audio_beams, step, **kwargs
            )
            
            if self.early_stopping and self._all_multimodal_finished(
                video_beams, audio_beams
            ):
                break
        
        return {
            'video': [max(beam, key=self._compute_final_score) for beam in video_beams],
            'audio': [max(beam, key=self._compute_final_score) for beam in audio_beams]
        }
    
    def _multimodal_step(
        self,
        model: torch.nn.Module,
        video_beams: List[List[BeamSearchHypothesis]],
        audio_beams: List[List[BeamSearchHypothesis]],
        step: int,
        **kwargs
    ) -> Tuple[List[List[BeamSearchHypothesis]], List[List[BeamSearchHypothesis]]]:
        """Perform one step of multimodal beam search."""
        batch_size = len(video_beams)
        
        new_video_beams = []
        new_audio_beams = []
        
        for batch_idx in range(batch_size):
            video_candidates = []
            audio_candidates = []
            
            # Generate candidates for each beam
            for beam_idx in range(len(video_beams[batch_idx])):
                video_hyp = video_beams[batch_idx][beam_idx]
                audio_hyp = audio_beams[batch_idx][beam_idx]
                
                if self._is_finished(video_hyp) and self._is_finished(audio_hyp):
                    video_candidates.append((video_hyp, video_hyp.score))
                    audio_candidates.append((audio_hyp, audio_hyp.score))
                    continue
                
                # Get multimodal predictions
                candidates = self._get_multimodal_candidates(
                    model, video_hyp, audio_hyp, **kwargs
                )
                
                video_candidates.extend(candidates['video'])
                audio_candidates.extend(candidates['audio'])
            
            # Select top candidates
            video_candidates.sort(key=lambda x: x[1], reverse=True)
            audio_candidates.sort(key=lambda x: x[1], reverse=True)
            
            new_video_beam = [cand[0] for cand in video_candidates[:self.beam_size]]
            new_audio_beam = [cand[0] for cand in audio_candidates[:self.beam_size]]
            
            new_video_beams.append(new_video_beam)
            new_audio_beams.append(new_audio_beam)
        
        return new_video_beams, new_audio_beams
    
    def _get_multimodal_candidates(
        self,
        model: torch.nn.Module,
        video_hyp: BeamSearchHypothesis,
        audio_hyp: BeamSearchHypothesis,
        **kwargs
    ) -> Dict[str, List[Tuple[BeamSearchHypothesis, float]]]:
        """Generate multimodal candidates with cross-modal consistency."""
        video_candidates = []
        audio_candidates = []
        
        with torch.no_grad():
            # Get model predictions for both modalities
            outputs = model.generate_multimodal_step(
                video=video_hyp.sequence,
                audio=audio_hyp.sequence,
                **kwargs
            )
            
            video_logits = outputs.get('video_logits')
            audio_logits = outputs.get('audio_logits')
            
            if video_logits is not None:
                video_log_probs = F.log_softmax(video_logits, dim=-1)
                top_video_probs, top_video_indices = torch.topk(
                    video_log_probs, k=self.beam_size, dim=-1
                )
                
                for i in range(self.beam_size):
                    # Create video candidate
                    token_id = top_video_indices[0, i]
                    log_prob = top_video_probs[0, i].item()
                    
                    new_sequence = torch.cat([
                        video_hyp.sequence,
                        token_id.unsqueeze(0).unsqueeze(0)
                    ], dim=1)
                    
                    new_hypothesis = BeamSearchHypothesis(
                        sequence=new_sequence,
                        score=video_hyp.score + log_prob * self.modality_weights['video'],
                        log_probs=video_hyp.log_probs + [log_prob]
                    )
                    
                    video_candidates.append((new_hypothesis, new_hypothesis.score))
            
            if audio_logits is not None:
                audio_log_probs = F.log_softmax(audio_logits, dim=-1)
                top_audio_probs, top_audio_indices = torch.topk(
                    audio_log_probs, k=self.beam_size, dim=-1
                )
                
                for i in range(self.beam_size):
                    # Create audio candidate
                    token_id = top_audio_indices[0, i]
                    log_prob = top_audio_probs[0, i].item()
                    
                    new_sequence = torch.cat([
                        audio_hyp.sequence,
                        token_id.unsqueeze(0).unsqueeze(0)
                    ], dim=1)
                    
                    new_hypothesis = BeamSearchHypothesis(
                        sequence=new_sequence,
                        score=audio_hyp.score + log_prob * self.modality_weights['audio'],
                        log_probs=audio_hyp.log_probs + [log_prob]
                    )
                    
                    audio_candidates.append((new_hypothesis, new_hypothesis.score))
            
        return {
            'video': video_candidates,
            'audio': audio_candidates
        }
    
    def _all_multimodal_finished(
        self,
        video_beams: List[List[BeamSearchHypothesis]],
        audio_beams: List[List[BeamSearchHypothesis]]
    ) -> bool:
        """Check if all multimodal beams are finished."""
        return (
            self._all_finished(video_beams) and 
            self._all_finished(audio_beams)
        ) 