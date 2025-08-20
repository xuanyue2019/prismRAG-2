"""
Data Quality Assessor for PrismRAG

Comprehensive data quality assessment and validation system that evaluates
generated data samples against multiple quality criteria and thresholds.
"""

import logging
import json
import time
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import statistics

from src.data_generation.evaluators import DistractorEvaluator, CoTEvaluator
from src.utils import ConfigManager, setup_logging


class QualityLevel(Enum):
    """Quality levels for data samples"""
    EXCELLENT = 4  # Meets all quality criteria with high scores
    GOOD = 3       # Meets most quality criteria
    FAIR = 2       # Meets basic quality criteria
    POOR = 1       # Fails to meet quality criteria
    REJECTED = 0   # Completely unacceptable


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for data samples"""
    relevance_score: float
    distraction_score: float
    format_score: float
    reasoning_score: float
    answer_score: float
    overall_score: float
    feedback: str
    quality_level: QualityLevel


@dataclass
class QualityThresholds:
    """Configurable quality thresholds for data filtering"""
    min_relevance: float = 3.0
    min_distraction: float = 3.0
    min_format: float = 3.0
    min_reasoning: float = 3.0
    min_answer: float = 3.0
    min_overall: float = 3.5
    max_iterations: int = 5


class DataQualityAssessor:
    """
    Comprehensive data quality assessment system for PrismRAG.
    
    Evaluates generated data samples against multiple quality criteria,
    applies configurable thresholds, and provides detailed feedback
    for iterative improvement.
    """
    
    def __init__(
        self,
        config: Optional[ConfigManager] = None,
        distractor_model: Optional[str] = None,
        cot_model: Optional[str] = None,
        device: str = "auto",
        quality_thresholds: Optional[QualityThresholds] = None
    ):
        """
        Initialize the data quality assessor.
        
        Args:
            config: Configuration manager instance
            distractor_model: Model for distractor evaluation
            cot_model: Model for CoT evaluation
            device: Device to run models on
            quality_thresholds: Custom quality thresholds
        """
        self.config = config or ConfigManager()
        self.distractor_model = distractor_model or self.config.get('model.base_model')
        self.cot_model = cot_model or self.config.get('model.base_model')
        self.device = device
        
        # Initialize evaluators
        self.distractor_evaluator = DistractorEvaluator(
            model_name=self.distractor_model,
            device=self.device
        )
        self.cot_evaluator = CoTEvaluator(
            model_name=self.cot_model,
            device=self.device
        )
        
        # Set quality thresholds
        self.thresholds = quality_thresholds or QualityThresholds()
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Statistics tracking
        self._evaluation_stats = {
            'total_samples': 0,
            'accepted_samples': 0,
            'rejected_samples': 0,
            'quality_distribution': {level.value: 0 for level in QualityLevel},
            'average_scores': {
                'relevance': 0.0,
                'distraction': 0.0,
                'format': 0.0,
                'reasoning': 0.0,
                'answer': 0.0,
                'overall': 0.0
            }
        }
        
        self.logger.info("Data quality assessor initialized")
    
    def assess_distractor_sample(
        self,
        question: str,
        answer: str,
        golden_passage: str,
        distractor_passage: str,
        open_ended_question: str,
        distractor_answer: str,
        user_time: Optional[str] = None,
        location: Optional[str] = None,
        max_attempts: int = 3
    ) -> Tuple[Optional[Dict], QualityMetrics]:
        """
        Assess quality of a distractor sample with iterative improvement.
        
        Args:
            question: Original question
            answer: Ground truth answer
            golden_passage: Original passage
            distractor_passage: Generated distractor passage
            open_ended_question: Modified open-ended question
            distractor_answer: Answer from distractor passage
            user_time: User time context
            location: User location context
            max_attempts: Maximum number of assessment attempts
            
        Returns:
            Tuple of (evaluation_dict, quality_metrics)
        """
        best_evaluation = None
        best_metrics = None
        best_score = 0.0
        
        for attempt in range(max_attempts):
            try:
                # Evaluate the distractor
                evaluation = self.distractor_evaluator.evaluate_distractor(
                    question=question,
                    answer=answer,
                    golden_passage=golden_passage,
                    distractor_passage=distractor_passage,
                    open_ended_question=open_ended_question,
                    distractor_answer=distractor_answer,
                    user_time=user_time,
                    location=location
                )
                
                # Calculate quality metrics
                metrics = self._calculate_quality_metrics(evaluation)
                
                # Update best result
                if metrics.overall_score > best_score:
                    best_evaluation = evaluation
                    best_metrics = metrics
                    best_score = metrics.overall_score
                
                # Check if quality meets threshold
                if self._meets_quality_threshold(metrics):
                    self.logger.debug(f"Distractor sample meets quality threshold on attempt {attempt + 1}")
                    break
                
                # Wait before next attempt (rate limiting)
                time.sleep(1)
                
            except Exception as e:
                self.logger.warning(f"Error in distractor assessment attempt {attempt + 1}: {e}")
                continue
        
        # Update statistics
        self._update_statistics(best_metrics)
        
        return best_evaluation, best_metrics
    
    def assess_cot_sample(
        self,
        question: str,
        references: List[str],
        strategy: str,
        reasoning: str,
        answer: str,
        ground_truth: str,
        max_attempts: int = 3
    ) -> Tuple[Optional[int], Optional[int], QualityMetrics]:
        """
        Assess quality of a CoT sample with iterative improvement.
        
        Args:
            question: The question being answered
            references: Reference documents
            strategy: Generated strategy
            reasoning: Generated reasoning chain
            answer: Generated answer
            ground_truth: Ground truth answer
            max_attempts: Maximum number of assessment attempts
            
        Returns:
            Tuple of (reasoning_score, answer_score, quality_metrics)
        """
        best_reasoning_score = 0
        best_answer_score = 0
        best_metrics = None
        
        for attempt in range(max_attempts):
            try:
                # Evaluate reasoning and answer
                reasoning_score = self.cot_evaluator.evaluate_reasoning(
                    question=question,
                    references=references,
                    strategy=strategy,
                    reasoning=reasoning,
                    answer=answer
                )
                
                answer_score = self.cot_evaluator.evaluate_answer(
                    question=question,
                    ground_truth=ground_truth,
                    candidate_answer=answer
                )
                
                # Create evaluation dict for metrics calculation
                evaluation = {
                    'reasoning_score': reasoning_score,
                    'answer_score': answer_score,
                    'overall_score': (reasoning_score + answer_score) / 2,
                    'feedback': f"Reasoning: {reasoning_score}, Answer: {answer_score}"
                }
                
                # Calculate quality metrics
                metrics = self._calculate_quality_metrics(evaluation)
                
                # Update best result
                current_score = metrics.overall_score
                if current_score > (best_reasoning_score + best_answer_score) / 2:
                    best_reasoning_score = reasoning_score
                    best_answer_score = answer_score
                    best_metrics = metrics
                
                # Check if quality meets threshold
                if self._meets_quality_threshold(metrics):
                    self.logger.debug(f"CoT sample meets quality threshold on attempt {attempt + 1}")
                    break
                
                # Wait before next attempt
                time.sleep(1)
                
            except Exception as e:
                self.logger.warning(f"Error in CoT assessment attempt {attempt + 1}: {e}")
                continue
        
        # Update statistics
        self._update_statistics(best_metrics)
        
        return best_reasoning_score, best_answer_score, best_metrics
    
    def _calculate_quality_metrics(self, evaluation: Dict) -> QualityMetrics:
        """Calculate comprehensive quality metrics from evaluation results"""
        # Extract scores with defaults
        relevance_score = evaluation.get('relevance_score', evaluation.get('relevance-score', 1.0))
        distraction_score = evaluation.get('distraction_score', evaluation.get('distraction-score', 1.0))
        format_score = evaluation.get('format_score', evaluation.get('format-score', 1.0))
        reasoning_score = evaluation.get('reasoning_score', 1.0)
        answer_score = evaluation.get('answer_score', 1.0)
        
        # Calculate overall score (weighted average)
        weights = {
            'relevance': 0.25,
            'distraction': 0.20,
            'format': 0.15,
            'reasoning': 0.20,
            'answer': 0.20
        }
        
        overall_score = (
            relevance_score * weights['relevance'] +
            distraction_score * weights['distraction'] +
            format_score * weights['format'] +
            reasoning_score * weights['reasoning'] +
            answer_score * weights['answer']
        )
        
        # Determine quality level
        if overall_score >= 3.8:
            quality_level = QualityLevel.EXCELLENT
        elif overall_score >= 3.3:
            quality_level = QualityLevel.GOOD
        elif overall_score >= 2.8:
            quality_level = QualityLevel.FAIR
        elif overall_score >= 2.0:
            quality_level = QualityLevel.POOR
        else:
            quality_level = QualityLevel.REJECTED
        
        return QualityMetrics(
            relevance_score=float(relevance_score),
            distraction_score=float(distraction_score),
            format_score=float(format_score),
            reasoning_score=float(reasoning_score),
            answer_score=float(answer_score),
            overall_score=float(overall_score),
            feedback=evaluation.get('feedback', ''),
            quality_level=quality_level
        )
    
    def _meets_quality_threshold(self, metrics: QualityMetrics) -> bool:
        """Check if metrics meet all quality thresholds"""
        return (
            metrics.relevance_score >= self.thresholds.min_relevance and
            metrics.distraction_score >= self.thresholds.min_distraction and
            metrics.format_score >= self.thresholds.min_format and
            metrics.reasoning_score >= self.thresholds.min_reasoning and
            metrics.answer_score >= self.thresholds.min_answer and
            metrics.overall_score >= self.thresholds.min_overall and
            metrics.quality_level.value >= QualityLevel.GOOD.value
        )
    
    def _update_statistics(self, metrics: Optional[QualityMetrics]) -> None:
        """Update evaluation statistics"""
        if not metrics:
            return
        
        self._evaluation_stats['total_samples'] += 1
        
        if self._meets_quality_threshold(metrics):
            self._evaluation_stats['accepted_samples'] += 1
        else:
            self._evaluation_stats['rejected_samples'] += 1
        
        # Update quality distribution
        self._evaluation_stats['quality_distribution'][metrics.quality_level.value] += 1
        
        # Update average scores (moving average)
        n = self._evaluation_stats['total_samples']
        current_avgs = self._evaluation_stats['average_scores']
        
        self._evaluation_stats['average_scores'] = {
            'relevance': (current_avgs['relevance'] * (n-1) + metrics.relevance_score) / n,
            'distraction': (current_avgs['distraction'] * (n-1) + metrics.distraction_score) / n,
            'format': (current_avgs['format'] * (n-1) + metrics.format_score) / n,
            'reasoning': (current_avgs['reasoning'] * (n-1) + metrics.reasoning_score) / n,
            'answer': (current_avgs['answer'] * (n-1) + metrics.answer_score) / n,
            'overall': (current_avgs['overall'] * (n-1) + metrics.overall_score) / n
        }
    
    def get_statistics(self) -> Dict:
        """Get current evaluation statistics"""
        return self._evaluation_stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset evaluation statistics"""
        self._evaluation_stats = {
            'total_samples': 0,
            'accepted_samples': 0,
            'rejected_samples': 0,
            'quality_distribution': {level.value: 0 for level in QualityLevel},
            'average_scores': {
                'relevance': 0.0,
                'distraction': 0.0,
                'format': 0.0,
                'reasoning': 0.0,
                'answer': 0.0,
                'overall': 0.0
            }
        }
    
    def assess_batch(
        self,
        samples: List[Dict],
        sample_type: str = "distractor",
        max_workers: int = 4
    ) -> List[Tuple[Dict, QualityMetrics]]:
        """
        Assess a batch of samples in parallel.
        
        Args:
            samples: List of sample dictionaries
            sample_type: Type of samples ("distractor" or "cot")
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of (sample, quality_metrics) tuples
        """
        results = []
        
        # Simple sequential processing for now (can be parallelized)
        for i, sample in enumerate(samples):
            self.logger.info(f"Assessing sample {i + 1}/{len(samples)}")
            
            try:
                if sample_type == "distractor":
                    evaluation, metrics = self.assess_distractor_sample(**sample)
                    results.append((sample, metrics))
                elif sample_type == "cot":
                    reasoning_score, answer_score, metrics = self.assess_cot_sample(**sample)
                    sample['reasoning_score'] = reasoning_score
                    sample['answer_score'] = answer_score
                    results.append((sample, metrics))
                else:
                    self.logger.warning(f"Unknown sample type: {sample_type}")
                    continue
                
            except Exception as e:
                self.logger.error(f"Error assessing sample {i + 1}: {e}")
                continue
        
        return results
    
    def filter_by_quality(
        self,
        samples_with_metrics: List[Tuple[Dict, QualityMetrics]],
        min_quality: QualityLevel = QualityLevel.GOOD
    ) -> List[Dict]:
        """
        Filter samples based on quality level.
        
        Args:
            samples_with_metrics: List of (sample, quality_metrics) tuples
            min_quality: Minimum quality level to accept
            
        Returns:
            List of filtered samples
        """
        filtered_samples = []
        
        for sample, metrics in samples_with_metrics:
            if metrics.quality_level.value >= min_quality.value:
                # Add quality metrics to sample for tracking
                sample['quality_metrics'] = {
                    'relevance_score': metrics.relevance_score,
                    'distraction_score': metrics.distraction_score,
                    'format_score': metrics.format_score,
                    'reasoning_score': metrics.reasoning_score,
                    'answer_score': metrics.answer_score,
                    'overall_score': metrics.overall_score,
                    'quality_level': metrics.quality_level.name,
                    'feedback': metrics.feedback
                }
                filtered_samples.append(sample)
        
        self.logger.info(
            f"Filtered {len(filtered_samples)}/{len(samples_with_metrics)} "
            f"samples with quality >= {min_quality.name}"
        )
        
        return filtered_samples
    
    def generate_quality_report(self) -> Dict:
        """Generate comprehensive quality assessment report"""
        stats = self.get_statistics()
        
        if stats['total_samples'] == 0:
            return {"message": "No samples assessed yet"}
        
        acceptance_rate = (stats['accepted_samples'] / stats['total_samples']) * 100
        
        report = {
            "total_samples_assessed": stats['total_samples'],
            "accepted_samples": stats['accepted_samples'],
            "rejected_samples": stats['rejected_samples'],
            "acceptance_rate": f"{acceptance_rate:.1f}%",
            "quality_distribution": stats['quality_distribution'],
            "average_scores": stats['average_scores'],
            "quality_thresholds": {
                "min_relevance": self.thresholds.min_relevance,
                "min_distraction": self.thresholds.min_distraction,
                "min_format": self.thresholds.min_format,
                "min_reasoning": self.thresholds.min_reasoning,
                "min_answer": self.thresholds.min_answer,
                "min_overall": self.thresholds.min_overall
            }
        }
        
        return report
    
    def save_quality_report(self, filepath: str) -> None:
        """Save quality report to JSON file"""
        report = self.generate_quality_report()
        
        from pathlib import Path
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Quality report saved to {filepath}")


# Example usage
if __name__ == "__main__":
    # Setup logging
    setup_logging()
    
    # Initialize quality assessor
    assessor = DataQualityAssessor()
    
    # Example distractor sample
    distractor_sample = {
        "question": "What is the capital of France?",
        "answer": "Paris",
        "golden_passage": "France is a country in Western Europe. Its capital is Paris, known for the Eiffel Tower.",
        "distractor_passage": "France is a nation in Western Europe. Its main city is Lyon, famous for its cuisine.",
        "open_ended_question": "Which city serves as the capital of France?",
        "distractor_answer": "Lyon",
        "user_time": "2024-01-15 14:30:00",
        "location": "Europe"
    }
    
    # Assess the sample
    evaluation, metrics = assessor.assess_distractor_sample(**distractor_sample)
    
    if evaluation:
        print(f"Distractor evaluation:")
        print(f"Relevance score: {metrics.relevance_score}")
        print(f"Distraction score: {metrics.distraction_score}")
        print(f"Format score: {metrics.format_score}")
        print(f"Overall score: {metrics.overall_score}")
        print(f"Quality level: {metrics.quality_level.name}")
        print(f"Feedback: {metrics.feedback[:100]}...")
    else:
        print("Failed to evaluate distractor sample")
    
    # Generate quality report
    report = assessor.generate_quality_report()
    print(f"\nQuality report: {json.dumps(report, indent=2)}")