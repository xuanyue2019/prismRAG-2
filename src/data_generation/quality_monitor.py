"""
Quality Monitor for PrismRAG

Real-time monitoring and feedback system for data generation quality,
providing continuous improvement through metrics tracking and analysis.
"""

import logging
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import statistics
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from src.data_generation.data_quality_assessor import DataQualityAssessor, QualityMetrics
from src.data_generation.data_validator import DataValidator, ValidationResult
from src.utils import setup_logging, ConfigManager


@dataclass
class QualityTrend:
    """Quality trend analysis results"""
    timestamp: datetime
    metric: str
    current_value: float
    trend: str  # "improving", "stable", "declining"
    change_percentage: float
    confidence: float


class QualityMonitor:
    """
    Real-time quality monitoring system for PrismRAG data generation.
    
    Tracks quality metrics over time, identifies trends, and provides
    feedback for continuous improvement of data generation processes.
    """
    
    def __init__(
        self,
        config: Optional[ConfigManager] = None,
        output_dir: str = "./quality_reports",
        update_interval: int = 100  # Update analysis every N samples
    ):
        """
        Initialize the quality monitor.
        
        Args:
            config: Configuration manager instance
            output_dir: Directory for quality reports and visualizations
            update_interval: Number of samples between trend analysis updates
        """
        self.config = config or ConfigManager()
        self.output_dir = Path(output_dir)
        self.update_interval = update_interval
        
        # Initialize components
        self.quality_assessor = DataQualityAssessor(config=config)
        self.data_validator = DataValidator()
        
        # Quality tracking
        self.quality_history: List[Dict] = []
        self.trend_analysis: Dict[str, QualityTrend] = {}
        self.last_analysis_time = datetime.now()
        
        # Statistics
        self.metrics_summary = {
            'total_samples': 0,
            'accepted_samples': 0,
            'rejected_samples': 0,
            'quality_distribution': {i: 0 for i in range(5)},  # 0-4 quality levels
            'average_scores': {
                'relevance': [],
                'distraction': [],
                'format': [],
                'reasoning': [],
                'answer': [],
                'overall': []
            }
        }
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("Quality monitor initialized")
    
    def track_sample_quality(
        self,
        sample: Dict,
        sample_type: str,
        metrics: Optional[QualityMetrics] = None,
        validation_result: Optional[ValidationResult] = None
    ) -> None:
        """
        Track quality metrics for a single sample.
        
        Args:
            sample: The data sample
            sample_type: Type of sample ("distractor", "cot", "seed_qa")
            metrics: Quality metrics from assessment
            validation_result: Validation results
        """
        timestamp = datetime.now()
        
        # Get quality metrics if not provided
        if metrics is None:
            metrics = self._assess_sample_quality(sample, sample_type)
        
        # Get validation results if not provided
        if validation_result is None:
            validation_result = self._validate_sample(sample, sample_type)
        
        # Create quality record
        quality_record = {
            'timestamp': timestamp.isoformat(),
            'sample_type': sample_type,
            'sample_id': self._generate_sample_id(sample),
            'metrics': {
                'relevance_score': metrics.relevance_score,
                'distraction_score': metrics.distraction_score,
                'format_score': metrics.format_score,
                'reasoning_score': metrics.reasoning_score,
                'answer_score': metrics.answer_score,
                'overall_score': metrics.overall_score,
                'quality_level': metrics.quality_level.value
            },
            'validation': {
                'is_valid': validation_result.is_valid,
                'error_count': len(validation_result.errors),
                'warning_count': len(validation_result.warnings),
                'suggestion_count': len(validation_result.suggestions)
            },
            'sample_metadata': self._extract_sample_metadata(sample)
        }
        
        # Add to history
        self.quality_history.append(quality_record)
        
        # Update statistics
        self._update_statistics(quality_record)
        
        # Periodic trend analysis
        if len(self.quality_history) % self.update_interval == 0:
            self._analyze_trends()
            self._generate_reports()
        
        self.logger.debug(f"Tracked quality for {sample_type} sample")
    
    def track_batch_quality(
        self,
        samples: List[Dict],
        sample_type: str,
        batch_id: Optional[str] = None
    ) -> Dict:
        """
        Track quality metrics for a batch of samples.
        
        Args:
            samples: List of data samples
            sample_type: Type of samples
            batch_id: Optional batch identifier
            
        Returns:
            Batch quality summary
        """
        batch_metrics = []
        batch_validation = []
        
        for sample in samples:
            metrics = self._assess_sample_quality(sample, sample_type)
            validation = self._validate_sample(sample, sample_type)
            
            self.track_sample_quality(
                sample=sample,
                sample_type=sample_type,
                metrics=metrics,
                validation_result=validation
            )
            
            batch_metrics.append(metrics)
            batch_validation.append(validation)
        
        # Generate batch summary
        batch_summary = self._generate_batch_summary(batch_metrics, batch_validation, batch_id)
        
        self.logger.info(
            f"Tracked quality for batch of {len(samples)} {sample_type} samples. "
            f"Acceptance rate: {batch_summary['acceptance_rate']}"
        )
        
        return batch_summary
    
    def _assess_sample_quality(self, sample: Dict, sample_type: str) -> QualityMetrics:
        """Assess sample quality using appropriate assessor"""
        try:
            if sample_type == "distractor":
                evaluation, metrics = self.quality_assessor.assess_distractor_sample(**sample)
            elif sample_type == "cot":
                _, _, metrics = self.quality_assessor.assess_cot_sample(**sample)
            else:
                # For seed_qa or other types, create basic metrics
                metrics = QualityMetrics(
                    relevance_score=3.0,
                    distraction_score=3.0,
                    format_score=3.0,
                    reasoning_score=3.0,
                    answer_score=3.0,
                    overall_score=3.0,
                    feedback="Basic assessment",
                    quality_level=self.quality_assessor._calculate_quality_level(3.0)
                )
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Error assessing sample quality: {e}")
            return QualityMetrics(
                relevance_score=1.0,
                distraction_score=1.0,
                format_score=1.0,
                reasoning_score=1.0,
                answer_score=1.0,
                overall_score=1.0,
                feedback=f"Assessment error: {e}",
                quality_level=self.quality_assessor._calculate_quality_level(1.0)
            )
    
    def _validate_sample(self, sample: Dict, sample_type: str) -> ValidationResult:
        """Validate sample using appropriate validator"""
        try:
            if sample_type == "distractor":
                return self.data_validator.validate_distractor_sample(sample)
            elif sample_type == "cot":
                return self.data_validator.validate_cot_sample(sample)
            elif sample_type == "seed_qa":
                return self.data_validator.validate_seed_qa_sample(sample)
            else:
                return ValidationResult(
                    is_valid=True,
                    validation_level=self.data_validator.ValidationLevel.INFO,
                    errors=[],
                    warnings=[],
                    suggestions=[]
                )
                
        except Exception as e:
            self.logger.warning(f"Error validating sample: {e}")
            return ValidationResult(
                is_valid=False,
                validation_level=self.data_validator.ValidationLevel.CRITICAL,
                errors=[f"Validation error: {e}"],
                warnings=[],
                suggestions=[]
            )
    
    def _generate_sample_id(self, sample: Dict) -> str:
        """Generate unique sample identifier"""
        import hashlib
        
        sample_str = json.dumps(sample, sort_keys=True)
        return hashlib.md5(sample_str.encode()).hexdigest()[:8]
    
    def _extract_sample_metadata(self, sample: Dict) -> Dict:
        """Extract relevant metadata from sample"""
        metadata = {}
        
        # Extract length information
        for field in ['question', 'answer', 'passage', 'reasoning', 'strategy']:
            if field in sample and isinstance(sample[field], str):
                metadata[f'{field}_length'] = len(sample[field])
                metadata[f'{field}_word_count'] = len(sample[field].split())
        
        # Extract quality metrics if present
        if 'quality_metrics' in sample and sample['quality_metrics']:
            metadata['has_quality_metrics'] = True
        
        return metadata
    
    def _update_statistics(self, quality_record: Dict) -> None:
        """Update monitoring statistics"""
        self.metrics_summary['total_samples'] += 1
        
        metrics = quality_record['metrics']
        validation = quality_record['validation']
        
        # Update acceptance count
        if validation['is_valid'] and metrics['overall_score'] >= 3.0:
            self.metrics_summary['accepted_samples'] += 1
        else:
            self.metrics_summary['rejected_samples'] += 1
        
        # Update quality distribution
        quality_level = metrics['quality_level']
        self.metrics_summary['quality_distribution'][quality_level] += 1
        
        # Update average scores
        for metric in ['relevance', 'distraction', 'format', 'reasoning', 'answer', 'overall']:
            score = metrics.get(f'{metric}_score', 0)
            self.metrics_summary['average_scores'][metric].append(score)
    
    def _analyze_trends(self) -> None:
        """Analyze quality trends over time"""
        if len(self.quality_history) < 10:  # Need sufficient data
            return
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(self.quality_history)
        
        # Extract metrics for trend analysis
        metrics_to_analyze = ['relevance_score', 'distraction_score', 'format_score', 
                             'reasoning_score', 'answer_score', 'overall_score']
        
        trends = {}
        
        for metric in metrics_to_analyze:
            # Extract metric values
            values = [item['metrics'][metric] for item in self.quality_history[-100:]]  # Last 100 samples
            
            if len(values) < 5:
                continue
            
            # Calculate trend
            current_value = values[-1]
            previous_value = values[-5] if len(values) >= 5 else values[0]
            
            change_percentage = ((current_value - previous_value) / previous_value * 100 
                               if previous_value != 0 else 0)
            
            # Determine trend direction
            if change_percentage > 5:
                trend = "improving"
                confidence = min(0.9, abs(change_percentage) / 20)
            elif change_percentage < -5:
                trend = "declining"
                confidence = min(0.9, abs(change_percentage) / 20)
            else:
                trend = "stable"
                confidence = 0.7
            
            trends[metric] = QualityTrend(
                timestamp=datetime.now(),
                metric=metric,
                current_value=current_value,
                trend=trend,
                change_percentage=change_percentage,
                confidence=confidence
            )
        
        self.trend_analysis = trends
        self.last_analysis_time = datetime.now()
        
        self.logger.info(f"Quality trends analyzed: {len(trends)} metrics")
    
    def _generate_batch_summary(
        self,
        batch_metrics: List[QualityMetrics],
        batch_validation: List[ValidationResult],
        batch_id: Optional[str] = None
    ) -> Dict:
        """Generate summary for a batch of samples"""
        if not batch_metrics:
            return {}
        
        # Calculate batch statistics
        overall_scores = [m.overall_score for m in batch_metrics]
        valid_samples = sum(1 for v in batch_validation if v.is_valid)
        
        summary = {
            'batch_id': batch_id or f"batch_{int(time.time())}",
            'sample_count': len(batch_metrics),
            'valid_samples': valid_samples,
            'invalid_samples': len(batch_metrics) - valid_samples,
            'acceptance_rate': f"{(valid_samples / len(batch_metrics) * 100):.1f}%",
            'average_overall_score': statistics.mean(overall_scores) if overall_scores else 0,
            'min_overall_score': min(overall_scores) if overall_scores else 0,
            'max_overall_score': max(overall_scores) if overall_scores else 0,
            'quality_distribution': {
                'excellent': sum(1 for m in batch_metrics if m.quality_level.value == 4),
                'good': sum(1 for m in batch_metrics if m.quality_level.value == 3),
                'fair': sum(1 for m in batch_metrics if m.quality_level.value == 2),
                'poor': sum(1 for m in batch_metrics if m.quality_level.value == 1),
                'rejected': sum(1 for m in batch_metrics if m.quality_level.value == 0)
            }
        }
        
        return summary
    
    def get_current_metrics(self) -> Dict:
        """Get current quality metrics summary"""
        return self.metrics_summary.copy()
    
    def get_trend_analysis(self) -> Dict[str, QualityTrend]:
        """Get current trend analysis"""
        return self.trend_analysis.copy()
    
    def generate_quality_report(self) -> Dict:
        """Generate comprehensive quality report"""
        stats = self.get_current_metrics()
        trends = self.get_trend_analysis()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary_statistics': stats,
            'trend_analysis': {
                metric: {
                    'current_value': trend.current_value,
                    'trend': trend.trend,
                    'change_percentage': trend.change_percentage,
                    'confidence': trend.confidence
                }
                for metric, trend in trends.items()
            },
            'sample_count': len(self.quality_history),
            'analysis_period': {
                'start': self.quality_history[0]['timestamp'] if self.quality_history else None,
                'end': self.quality_history[-1]['timestamp'] if self.quality_history else None,
                'duration_hours': (
                    (datetime.fromisoformat(self.quality_history[-1]['timestamp']) -
                    datetime.fromisoformat(self.quality_history[0]['timestamp'])
                ).total_seconds() / 3600 if len(self.quality_history) >= 2 else 0
                )
            }
        }
        
        return report
    
    def save_quality_report(self, filename: Optional[str] = None) -> str:
        """Save quality report to file"""
        report = self.generate_quality_report()
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"quality_report_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Quality report saved to {filepath}")
        return str(filepath)
    
    def visualize_quality_trends(self) -> None:
        """Create visualizations of quality trends"""
        if len(self.quality_history) < 5:
            self.logger.warning("Insufficient data for visualization")
            return
        
        # Create trend plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('PrismRAG Data Quality Trends', fontsize=16)
        
        metrics = ['overall_score', 'relevance_score', 'distraction_score', 
                  'format_score', 'reasoning_score', 'answer_score']
        titles = ['Overall', 'Relevance', 'Distraction', 'Format', 'Reasoning', 'Answer']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i//3, i%3]
            
            # Extract metric values
            values = [item['metrics'][metric] for item in self.quality_history]
            timestamps = [datetime.fromisoformat(item['timestamp']) for item in self.quality_history]
            
            ax.plot(timestamps, values, 'b-', alpha=0.7)
            ax.set_title(f'{title} Score Trend')
            ax.set_ylabel('Score')
            ax.set_ylim(1, 5)
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            if len(values) > 10:
                z = np.polyfit(range(len(values)), values, 1)
                p = np.poly1d(z)
                ax.plot(timestamps, p(range(len(values))), 'r--', alpha=0.8)
        
        plt.tight_layout()
        
        # Save visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_path = self.output_dir / f"quality_trends_{timestamp}.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Quality visualization saved to {viz_path}")
    
    def get_quality_feedback(self) -> List[str]:
        """Generate actionable feedback based on quality analysis"""
        feedback = []
        
        # Check overall quality trend
        overall_trend = self.trend_analysis.get('overall_score')
        if overall_trend and overall_trend.trend == "declining":
            feedback.append(
                f"Overall quality is declining ({overall_trend.change_percentage:.1f}%). "
                "Consider reviewing recent generation parameters."
            )
        
        # Check individual metric trends
        for metric, trend in self.trend_analysis.items():
            if trend.trend == "declining" and trend.confidence > 0.6:
                feedback.append(
                    f"{metric.replace('_', ' ').title()} quality is declining "
                    f"({trend.change_percentage:.1f}%). May need attention."
                )
        
        # Check acceptance rate
        acceptance_rate = (self.metrics_summary['accepted_samples'] /
                          self.metrics_summary['total_samples'] * 100)
        if acceptance_rate < 70:
            feedback.append(
                f"Low acceptance rate ({acceptance_rate:.1f}%). "
                "Consider adjusting quality thresholds or improving generation."
            )
        
        # Check quality distribution
        excellent_count = self.metrics_summary['quality_distribution'].get(4, 0)
        if excellent_count / self.metrics_summary['total_samples'] < 0.2:
            feedback.append(
                "Low proportion of excellent quality samples. "
                "Consider optimizing generation strategies."
            )
        
        return feedback
    
    def reset_monitoring(self) -> None:
        """Reset all monitoring data"""
        self.quality_history = []
        self.trend_analysis = {}
        self.metrics_summary = {
            'total_samples': 0,
            'accepted_samples': 0,
            'rejected_samples': 0,
            'quality_distribution': {i: 0 for i in range(5)},
            'average_scores': {
                'relevance': [],
                'distraction': [],
                'format': [],
                'reasoning': [],
                'answer': [],
                'overall': []
            }
        }
        
        self.logger.info("Quality monitoring data reset")


# Example usage
if __name__ == "__main__":
    # Setup logging
    setup_logging()
    
    # Initialize quality monitor
    monitor = QualityMonitor()
    
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
    
    # Track sample quality
    monitor.track_sample_quality(distractor_sample, "distractor")
    
    # Get current metrics
    metrics = monitor.get_current_metrics()
    print(f"Total samples tracked: {metrics['total_samples']}")
    print(f"Accepted samples: {metrics['accepted_samples']}")
    
    # Generate feedback
    feedback = monitor.get_quality_feedback()
    if feedback:
        print("\nQuality feedback:")
        for item in feedback:
            print(f"- {item}")
    else:
        print("\nNo significant quality issues detected")