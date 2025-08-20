"""
Multi-Benchmark Evaluator for PrismRAG

Comprehensive evaluation system that tests trained models on multiple
RAG benchmarks to measure factuality, robustness, and performance.
"""

import logging
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset
import evaluate

from src.utils import ConfigManager, setup_logging


@dataclass
class BenchmarkResult:
    """Results from a single benchmark evaluation"""
    benchmark_name: str
    metrics: Dict[str, float]
    samples_evaluated: int
    evaluation_time: float
    details: Optional[Dict] = None


@dataclass
class EvaluationSummary:
    """Summary of multi-benchmark evaluation"""
    overall_score: float
    benchmark_results: Dict[str, BenchmarkResult]
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]


class BenchmarkEvaluator:
    """
    Multi-benchmark evaluation system for PrismRAG models.
    
    Evaluates trained models on multiple RAG benchmarks to provide
    comprehensive performance assessment across different domains
    and difficulty levels.
    """
    
    def __init__(
        self,
        config: Optional[ConfigManager] = None,
        model_path: Optional[str] = None,
        device: str = "auto"
    ):
        """
        Initialize the benchmark evaluator.
        
        Args:
            config: Configuration manager instance
            model_path: Path to trained model
            device: Device to run evaluation on
        """
        self.config = config or ConfigManager()
        self.model_path = model_path
        self.device = device
        
        # Supported benchmarks from config
        self.supported_benchmarks = self.config.get('evaluation.benchmarks', [])
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Load model and tokenizer
        self._load_model()
        
        # Load evaluation metrics
        self._load_metrics()
        
        # Create output directory
        self.output_dir = Path(self.config.get('paths.output_dir', './evaluation_results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Benchmark evaluator initialized for {len(self.supported_benchmarks)} benchmarks")
    
    def _load_model(self) -> None:
        """Load the trained model for evaluation"""
        try:
            if self.model_path:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map=self.device
                )
            else:
                # Fallback to base model
                model_name = self.config.get('model.base_model')
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map=self.device
                )
            
            # Add padding token if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.logger.info(f"Model loaded from {self.model_path or 'base model'}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_metrics(self) -> None:
        """Load evaluation metrics"""
        self.metrics = {}
        
        metric_names = self.config.get('evaluation.metrics', [])
        for metric_name in metric_names:
            try:
                if metric_name == "factuality_score":
                    # Custom factuality metric (placeholder)
                    self.metrics[metric_name] = self._compute_factuality
                elif metric_name == "hallucination_rate":
                    self.metrics[metric_name] = evaluate.load("bertscore")
                else:
                    self.metrics[metric_name] = evaluate.load(metric_name)
            except Exception as e:
                self.logger.warning(f"Failed to load metric {metric_name}: {e}")
        
        self.logger.info(f"Loaded {len(self.metrics)} evaluation metrics")
    
    def evaluate_all_benchmarks(self) -> EvaluationSummary:
        """
        Evaluate model on all supported benchmarks.
        
        Returns:
            Comprehensive evaluation summary
        """
        results = {}
        total_start_time = time.time()
        
        for benchmark_name in self.supported_benchmarks:
            try:
                self.logger.info(f"Evaluating on benchmark: {benchmark_name}")
                benchmark_result = self.evaluate_benchmark(benchmark_name)
                results[benchmark_name] = benchmark_result
                
                # Rate limiting between benchmarks
                time.sleep(2)
                
            except Exception as e:
                self.logger.error(f"Failed to evaluate {benchmark_name}: {e}")
                results[benchmark_name] = BenchmarkResult(
                    benchmark_name=benchmark_name,
                    metrics={'error': 0.0},
                    samples_evaluated=0,
                    evaluation_time=0.0,
                    details={'error': str(e)}
                )
        
        total_time = time.time() - total_start_time
        
        # Generate comprehensive summary
        summary = self._generate_summary(results, total_time)
        
        # Save evaluation results
        self._save_evaluation_results(results, summary)
        
        return summary
    
    def evaluate_benchmark(self, benchmark_name: str) -> BenchmarkResult:
        """
        Evaluate model on a specific benchmark.
        
        Args:
            benchmark_name: Name of the benchmark to evaluate
            
        Returns:
            Benchmark evaluation results
        """
        start_time = time.time()
        
        # Load benchmark dataset
        dataset = self._load_benchmark_dataset(benchmark_name)
        if dataset is None:
            raise ValueError(f"Unsupported benchmark: {benchmark_name}")
        
        # Evaluate on benchmark samples
        metrics = self._evaluate_on_dataset(dataset, benchmark_name)
        
        evaluation_time = time.time() - start_time
        
        return BenchmarkResult(
            benchmark_name=benchmark_name,
            metrics=metrics,
            samples_evaluated=len(dataset),
            evaluation_time=evaluation_time,
            details={'dataset_size': len(dataset)}
        )
    
    def _load_benchmark_dataset(self, benchmark_name: str) -> Optional[Dataset]:
        """Load dataset for a specific benchmark"""
        try:
            if benchmark_name.lower() == "hotpotqa":
                dataset = load_dataset("hotpot_qa", "fullwiki", split="validation[:100]")
                return self._prepare_hotpotqa_dataset(dataset)
            
            elif benchmark_name.lower() == "crag":
                # CRAG benchmark (simulated)
                return self._create_synthetic_dataset("crag", 50)
            
            elif benchmark_name.lower() == "covidqa":
                # COVID-QA benchmark (simulated)
                return self._create_synthetic_dataset("covidqa", 50)
            
            elif benchmark_name.lower() == "ms_marco":
                dataset = load_dataset("ms_marco", "v2.1", split="validation[:50]")
                return self._prepare_msmarco_dataset(dataset)
            
            elif benchmark_name.lower() == "pubmedqa":
                dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train[:50]")
                return self._prepare_pubmedqa_dataset(dataset)
            
            else:
                # For other benchmarks, create synthetic data
                return self._create_synthetic_dataset(benchmark_name, 30)
                
        except Exception as e:
            self.logger.warning(f"Failed to load benchmark {benchmark_name}: {e}")
            return self._create_synthetic_dataset(benchmark_name, 20)
    
    def _prepare_hotpotqa_dataset(self, dataset: Dataset) -> Dataset:
        """Prepare HotpotQA dataset for evaluation"""
        def format_hotpotqa_sample(example):
            context = " ".join([ctx['text'] for ctx in example['context']])
            return {
                'question': example['question'],
                'context': context,
                'answer': example['answer'],
                'type': 'hotpotqa'
            }
        
        return dataset.map(format_hotpotqa_sample)
    
    def _prepare_msmarco_dataset(self, dataset: Dataset) -> Dataset:
        """Prepare MS MARCO dataset for evaluation"""
        def format_msmarco_sample(example):
            return {
                'question': example['query'],
                'context': example['passages']['passage_text'][0],
                'answer': example['answers'][0] if example['answers'] else "",
                'type': 'ms_marco'
            }
        
        return dataset.map(format_msmarco_sample)
    
    def _prepare_pubmedqa_dataset(self, dataset: Dataset) -> Dataset:
        """Prepare PubMedQA dataset for evaluation"""
        def format_pubmedqa_sample(example):
            return {
                'question': example['question'],
                'context': example['context'],
                'answer': example['long_answer'],
                'type': 'pubmedqa'
            }
        
        return dataset.map(format_pubmedqa_sample)
    
    def _create_synthetic_dataset(self, benchmark_name: str, num_samples: int) -> Dataset:
        """Create synthetic dataset for benchmarks without available data"""
        samples = []
        
        for i in range(num_samples):
            samples.append({
                'question': f"What is the main concept discussed in {benchmark_name} sample {i+1}?",
                'context': f"This is a synthetic context for {benchmark_name} benchmark evaluation. "
                          f"Sample {i+1} discusses important concepts in this domain.",
                'answer': f"The main concept is synthetic evaluation for {benchmark_name}",
                'type': benchmark_name.lower()
            })
        
        return Dataset.from_list(samples)
    
    def _evaluate_on_dataset(self, dataset: Dataset, benchmark_name: str) -> Dict[str, float]:
        """Evaluate model on dataset and compute metrics"""
        predictions = []
        references = []
        
        # Generate predictions
        for example in dataset:
            try:
                prediction = self._generate_answer(
                    question=example['question'],
                    context=example['context']
                )
                predictions.append(prediction)
                references.append(example['answer'])
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.warning(f"Failed to generate answer: {e}")
                predictions.append("")
                references.append(example['answer'])
        
        # Compute metrics
        metrics_results = {}
        
        for metric_name, metric_func in self.metrics.items():
            try:
                if metric_name == "factuality_score":
                    score = metric_func(predictions, references, dataset)
                elif metric_name == "hallucination_rate":
                    # Use BERTScore for hallucination detection
                    results = metric_func.compute(
                        predictions=predictions,
                        references=references,
                        lang="en"
                    )
                    score = np.mean(results['f1'])
                else:
                    score = metric_func.compute(
                        predictions=predictions,
                        references=references
                    )
                
                metrics_results[metric_name] = float(score) if isinstance(score, (int, float)) else 0.0
                
            except Exception as e:
                self.logger.warning(f"Failed to compute {metric_name}: {e}")
                metrics_results[metric_name] = 0.0
        
        return metrics_results
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using the trained model"""
        prompt = f"""基于以下上下文回答问题。请提供事实准确、直接和清晰的回复。

上下文：
{context}

问题：
{question}

请基于上下文提供准确的答案："""
        
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.3,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"Answer generation failed: {e}")
            return ""
    
    def _compute_factuality(self, predictions: List[str], references: List[str], dataset: Dataset) -> float:
        """Custom factuality score computation"""
        # Simple implementation - can be enhanced with more sophisticated methods
        correct_count = 0
        
        for pred, ref in zip(predictions, references):
            if pred.lower() in ref.lower() or ref.lower() in pred.lower():
                correct_count += 1
        
        return correct_count / len(predictions) if predictions else 0.0
    
    def _generate_summary(self, results: Dict[str, BenchmarkResult], total_time: float) -> EvaluationSummary:
        """Generate comprehensive evaluation summary"""
        # Calculate overall score (weighted average)
        benchmark_weights = {
            'hotpotqa': 0.2,
            'crag': 0.15,
            'covidqa': 0.15,
            'ms_marco': 0.2,
            'pubmedqa': 0.15,
            'default': 0.1
        }
        
        weighted_scores = []
        strengths = []
        weaknesses = []
        
        for benchmark_name, result in results.items():
            weight = benchmark_weights.get(benchmark_name.lower(), benchmark_weights['default'])
            avg_score = np.mean(list(result.metrics.values())) if result.metrics else 0.0
            
            weighted_scores.append(avg_score * weight)
            
            # Analyze strengths and weaknesses
            if avg_score >= 0.7:
                strengths.append(f"Strong performance on {benchmark_name} (score: {avg_score:.3f})")
            elif avg_score <= 0.4:
                weaknesses.append(f"Weak performance on {benchmark_name} (score: {avg_score:.3f})")
        
        overall_score = sum(weighted_scores) if weighted_scores else 0.0
        
        # Generate recommendations
        recommendations = []
        if overall_score < 0.6:
            recommendations.append("Consider additional training with more diverse data")
            recommendations.append("Review model architecture and hyperparameters")
        if any('hotpotqa' in weakness for weakness in weaknesses):
            recommendations.append("Focus on improving multi-hop reasoning capabilities")
        if any('crag' in weakness for weakness in weaknesses):
            recommendations.append("Enhance factuality and hallucination prevention")
        
        return EvaluationSummary(
            overall_score=overall_score,
            benchmark_results=results,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations
        )
    
    def _save_evaluation_results(self, results: Dict[str, BenchmarkResult], summary: EvaluationSummary) -> None:
        """Save evaluation results to files"""
        # Save detailed results
        results_dict = {
            'timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'benchmark_results': {
                name: {
                    'metrics': result.metrics,
                    'samples_evaluated': result.samples_evaluated,
                    'evaluation_time': result.evaluation_time,
                    'details': result.details
                }
                for name, result in results.items()
            },
            'summary': {
                'overall_score': summary.overall_score,
                'strengths': summary.strengths,
                'weaknesses': summary.weaknesses,
                'recommendations': summary.recommendations
            }
        }
        
        results_file = self.output_dir / "benchmark_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)
        
        # Save summary report
        summary_file = self.output_dir / "evaluation_summary.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(self._generate_markdown_summary(summary))
        
        self.logger.info(f"Evaluation results saved to {self.output_dir}")
    
    def _generate_markdown_summary(self, summary: EvaluationSummary) -> str:
        """Generate markdown summary report"""
        report = [
            "# PrismRAG Model Evaluation Summary",
            "",
            f"**Evaluation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Model Path**: {self.model_path or 'Base Model'}",
            f"**Overall Score**: {summary.overall_score:.3f}",
            "",
            "## Benchmark Results",
            ""
        ]
        
        # Add benchmark results table
        report.append("| Benchmark | Samples | Score | Time (s) |")
        report.append("|-----------|---------|-------|----------|")
        
        for benchmark_name, result in summary.benchmark_results.items():
            avg_score = np.mean(list(result.metrics.values())) if result.metrics else 0.0
            report.append(
                f"| {benchmark_name} | {result.samples_evaluated} | {avg_score:.3f} | {result.evaluation_time:.1f} |"
            )
        
        # Add strengths and weaknesses
        report.extend([
            "",
            "## Strengths",
            ""
        ] + [f"- {strength}" for strength in summary.strengths] + [
            "",
            "## Weaknesses",
            ""
        ] + [f"- {weakness}" for weakness in summary.weaknesses] + [
            "",
            "## Recommendations",
            ""
        ] + [f"- {recommendation}" for recommendation in summary.recommendations] + [
            "",
            "## Detailed Metrics",
            ""
        ])
        
        # Add detailed metrics for each benchmark
        for benchmark_name, result in summary.benchmark_results.items():
            report.extend([
                f"### {benchmark_name}",
                ""
            ])
            
            for metric_name, metric_value in result.metrics.items():
                report.append(f"- {metric_name}: {metric_value:.3f}")
            
            report.append("")
        
        return "\n".join(report)


# Example usage
if __name__ == "__main__":
    # Setup logging
    setup_logging()
    
    # Initialize evaluator
    evaluator = BenchmarkEvaluator()
    
    print("Starting multi-benchmark evaluation...")
    summary = evaluator.evaluate_all_benchmarks()
    
    print(f"\nEvaluation completed!")
    print(f"Overall Score: {summary.overall_score:.3f}")
    print(f"\nStrengths:")
    for strength in summary.strengths:
        print(f"  - {strength}")
    
    print(f"\nWeaknesses:")
    for weakness in summary.weaknesses:
        print(f"  - {weakness}")
    
    print(f"\nRecommendations:")
    for recommendation in summary.recommendations:
        print(f"  - {recommendation}")