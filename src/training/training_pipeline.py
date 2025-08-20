"""
End-to-End Training Pipeline for PrismRAG

Complete training pipeline that integrates data generation, quality assessment,
and model training into a unified workflow.
"""

import logging
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, concatenate_datasets

from src.data_acquisition.wikipedia_client import WikipediaClient
from src.data_acquisition.web_search_client import WebSearchClient
from src.data_generation.seed_data_generator import SeedDataGenerator
from src.data_generation.distractor_generator import DistractorGenerator
from src.data_generation.strategic_cot_generator import StrategicCoTGenerator
from src.data_generation.data_quality_assessor import DataQualityAssessor
from src.data_generation.data_validator import DataValidator
from src.data_generation.quality_monitor import QualityMonitor
from src.utils import ConfigManager, setup_logging


@dataclass
class TrainingPipelineConfig:
    """Configuration for the training pipeline"""
    # Data generation
    num_seed_samples: int = 1000
    num_distractor_samples: int = 2000
    num_cot_samples: int = 1500
    
    # Quality thresholds
    min_quality_score: float = 3.5
    min_acceptance_rate: float = 0.7
    
    # Training parameters
    batch_size: int = 4
    learning_rate: float = 2e-5
    num_epochs: int = 3
    max_seq_length: int = 2048
    
    # Paths
    output_dir: str = "./training_output"
    data_cache_dir: str = "./data_cache"
    model_output_dir: str = "./models"


class TrainingPipeline:
    """
    End-to-end training pipeline for PrismRAG.
    
    Integrates data acquisition, generation, quality assessment,
    and model training into a complete workflow.
    """
    
    def __init__(
        self,
        config: Optional[TrainingPipelineConfig] = None,
        model_name: str = "meta-llama/Llama-3.1-70b-instruct"
    ):
        """
        Initialize the training pipeline.
        
        Args:
            config: Pipeline configuration
            model_name: Base model name for training
        """
        self.config = config or TrainingPipelineConfig()
        self.model_name = model_name
        
        # Initialize components
        self.config_manager = ConfigManager()
        self.setup_logging()
        
        # Data acquisition
        self.wikipedia_client = WikipediaClient()
        self.web_search_client = WebSearchClient()
        
        # Data generation
        self.seed_generator = SeedDataGenerator()
        self.distractor_generator = DistractorGenerator()
        self.cot_generator = StrategicCoTGenerator()
        
        # Quality management
        self.quality_assessor = DataQualityAssessor()
        self.data_validator = DataValidator()
        self.quality_monitor = QualityMonitor()
        
        # Training components
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        # Create output directories
        self._create_directories()
        
        self.logger.info("Training pipeline initialized")
    
    def setup_logging(self) -> None:
        """Setup logging for the pipeline"""
        setup_logging()
        self.logger = logging.getLogger(__name__)
    
    def _create_directories(self) -> None:
        """Create necessary output directories"""
        directories = [
            self.config.output_dir,
            self.config.data_cache_dir,
            self.config.model_output_dir,
            f"{self.config.output_dir}/reports",
            f"{self.config.output_dir}/visualizations"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def run_pipeline(self) -> Dict:
        """
        Run the complete training pipeline.
        
        Returns:
            Pipeline execution results
        """
        pipeline_start = datetime.now()
        results = {}
        
        try:
            # Phase 1: Data Acquisition
            self.logger.info("=== Phase 1: Data Acquisition ===")
            acquisition_results = self._acquire_training_data()
            results['acquisition'] = acquisition_results
            
            # Phase 2: Data Generation
            self.logger.info("=== Phase 2: Data Generation ===")
            generation_results = self._generate_training_data(acquisition_results)
            results['generation'] = generation_results
            
            # Phase 3: Quality Assessment
            self.logger.info("=== Phase 3: Quality Assessment ===")
            quality_results = self._assess_data_quality(generation_results)
            results['quality'] = quality_results
            
            # Phase 4: Data Preparation
            self.logger.info("=== Phase 4: Data Preparation ===")
            preparation_results = self._prepare_training_data(quality_results)
            results['preparation'] = preparation_results
            
            # Phase 5: Model Training
            self.logger.info("=== Phase 5: Model Training ===")
            training_results = self._train_model(preparation_results)
            results['training'] = training_results
            
            # Phase 6: Evaluation
            self.logger.info("=== Phase 6: Model Evaluation ===")
            evaluation_results = self._evaluate_model(training_results)
            results['evaluation'] = evaluation_results
            
            # Generate final report
            pipeline_end = datetime.now()
            results['summary'] = {
                'start_time': pipeline_start.isoformat(),
                'end_time': pipeline_end.isoformat(),
                'duration_seconds': (pipeline_end - pipeline_start).total_seconds(),
                'success': True,
                'total_samples': sum([
                    generation_results.get('seed_samples', 0),
                    generation_results.get('distractor_samples', 0),
                    generation_results.get('cot_samples', 0)
                ])
            }
            
            self.logger.info("Training pipeline completed successfully")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            results['summary'] = {
                'success': False,
                'error': str(e),
                'end_time': datetime.now().isoformat()
            }
        
        # Save pipeline results
        self._save_pipeline_results(results)
        
        return results
    
    def _acquire_training_data(self) -> Dict:
        """Acquire raw training data from various sources"""
        results = {
            'wikipedia_articles': [],
            'web_search_results': [],
            'acquisition_time': datetime.now().isoformat()
        }
        
        try:
            # Acquire Wikipedia articles
            topics = [
                "artificial intelligence", "machine learning", "natural language processing",
                "computer science", "physics", "biology", "history", "geography"
            ]
            
            for topic in topics:
                self.logger.info(f"Acquiring Wikipedia articles for: {topic}")
                articles = self.wikipedia_client.search_and_extract(topic, max_articles=3)
                results['wikipedia_articles'].extend(articles)
                time.sleep(1)  # Rate limiting
            
            # Acquire web search results
            search_queries = [
                "recent AI developments", "machine learning applications",
                "NLP research papers", "computer science fundamentals"
            ]
            
            for query in search_queries:
                self.logger.info(f"Searching web for: {query}")
                search_results = self.web_search_client.search(query, max_results=5)
                results['web_search_results'].extend(search_results)
                time.sleep(2)  # Rate limiting
            
            self.logger.info(f"Acquired {len(results['wikipedia_articles'])} Wikipedia articles "
                           f"and {len(results['web_search_results'])} web search results")
            
        except Exception as e:
            self.logger.warning(f"Data acquisition partially failed: {e}")
        
        return results
    
    def _generate_training_data(self, acquisition_results: Dict) -> Dict:
        """Generate training data from acquired content"""
        results = {
            'seed_samples': [],
            'distractor_samples': [],
            'cot_samples': [],
            'generation_time': datetime.now().isoformat()
        }
        
        # Combine all acquired content
        all_content = []
        all_content.extend(acquisition_results['wikipedia_articles'])
        all_content.extend([result['content'] for result in acquisition_results['web_search_results']])
        
        if not all_content:
            self.logger.warning("No content available for data generation")
            return results
        
        # Generate seed QA samples
        self.logger.info("Generating seed QA samples...")
        seed_samples = self.seed_generator.generate_batch(
            documents=all_content,
            num_samples=self.config.num_seed_samples,
            max_attempts=3
        )
        results['seed_samples'] = seed_samples
        
        # Generate distractor samples
        self.logger.info("Generating distractor samples...")
        distractor_samples = []
        for seed_sample in seed_samples[:self.config.num_distractor_samples]:
            try:
                distractor_sample = self.distractor_generator.generate_distractor(**seed_sample)
                if distractor_sample:
                    distractor_samples.append(distractor_sample)
            except Exception as e:
                self.logger.warning(f"Failed to generate distractor: {e}")
        
        results['distractor_samples'] = distractor_samples
        
        # Generate CoT samples
        self.logger.info("Generating strategic CoT samples...")
        cot_samples = []
        cot_inputs = []
        
        # Prepare inputs for CoT generation
        for seed_sample in seed_samples[:self.config.num_cot_samples]:
            cot_inputs.append({
                'question': seed_sample['question'],
                'references': [seed_sample['passage']],
                'ground_truth_answer': seed_sample['answer']
            })
        
        # Generate CoT in batches
        batch_size = 10
        for i in range(0, len(cot_inputs), batch_size):
            batch = cot_inputs[i:i + batch_size]
            batch_cot_samples = self.cot_generator.generate_batch(batch)
            cot_samples.extend(batch_cot_samples)
            time.sleep(1)  # Rate limiting
        
        results['cot_samples'] = cot_samples
        
        self.logger.info(f"Generated {len(seed_samples)} seed, {len(distractor_samples)} distractor, "
                       f"and {len(cot_samples)} CoT samples")
        
        # Save generated data
        self._save_generated_data(results)
        
        return results
    
    def _assess_data_quality(self, generation_results: Dict) -> Dict:
        """Assess quality of generated data"""
        results = {
            'assessed_samples': [],
            'quality_reports': [],
            'assessment_time': datetime.now().isoformat()
        }
        
        # Assess seed samples
        seed_quality = self.quality_monitor.track_batch(
            generation_results['seed_samples'],
            sample_type="seed_qa",
            batch_id="seed_batch_1"
        )
        results['quality_reports'].append(seed_quality)
        
        # Assess distractor samples
        distractor_quality = self.quality_monitor.track_batch(
            generation_results['distractor_samples'],
            sample_type="distractor",
            batch_id="distractor_batch_1"
        )
        results['quality_reports'].append(distractor_quality)
        
        # Assess CoT samples
        cot_quality = self.quality_monitor.track_batch(
            generation_results['cot_samples'],
            sample_type="cot",
            batch_id="cot_batch_1"
        )
        results['quality_reports'].append(cot_quality)
        
        # Generate overall quality report
        overall_report = self.quality_monitor.generate_quality_report()
        results['overall_quality'] = overall_report
        
        # Save quality reports
        self._save_quality_reports(results)
        
        # Check if quality meets thresholds
        acceptance_rates = [report.get('acceptance_rate', '0%') for report in results['quality_reports']]
        avg_acceptance = sum(float(rate.strip('%')) for rate in acceptance_rates) / len(acceptance_rates)
        
        if avg_acceptance < self.config.min_acceptance_rate * 100:
            self.logger.warning(f"Low acceptance rate: {avg_acceptance:.1f}% "
                              f"(threshold: {self.config.min_acceptance_rate * 100:.1f}%)")
        
        return results
    
    def _prepare_training_data(self, quality_results: Dict) -> Dict:
        """Prepare data for model training"""
        results = {
            'training_dataset': None,
            'validation_dataset': None,
            'preparation_time': datetime.now().isoformat()
        }
        
        # Combine all quality-assessed samples
        all_samples = []
        
        # Add seed samples (convert to training format)
        for sample in quality_results.get('seed_samples', []):
            training_sample = self._convert_to_training_format(sample, sample_type="seed_qa")
            if training_sample:
                all_samples.append(training_sample)
        
        # Add distractor samples
        for sample in quality_results.get('distractor_samples', []):
            training_sample = self._convert_to_training_format(sample, sample_type="distractor")
            if training_sample:
                all_samples.append(training_sample)
        
        # Add CoT samples
        for sample in quality_results.get('cot_samples', []):
            training_sample = self._convert_to_training_format(sample, sample_type="cot")
            if training_sample:
                all_samples.append(training_sample)
        
        if not all_samples:
            raise ValueError("No training samples available after quality assessment")
        
        # Create datasets
        dataset = Dataset.from_list(all_samples)
        
        # Split into train/validation
        train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
        
        results['training_dataset'] = train_test_split['train']
        results['validation_dataset'] = train_test_split['test']
        
        self.logger.info(f"Prepared {len(all_samples)} samples for training "
                       f"({len(results['training_dataset'])} train, "
                       f"{len(results['validation_dataset'])} validation)")
        
        # Save prepared datasets
        self._save_datasets(results)
        
        return results
    
    def _convert_to_training_format(self, sample: Dict, sample_type: str) -> Optional[Dict]:
        """Convert sample to model training format"""
        try:
            if sample_type == "seed_qa":
                return {
                    'text': f"Question: {sample['question']}\nAnswer: {sample['answer']}\nContext: {sample['passage']}",
                    'labels': sample['answer']
                }
            
            elif sample_type == "distractor":
                return {
                    'text': (f"Question: {sample['open_ended_question']}\n"
                            f"Distractor Answer: {sample['distractor_answer']}\n"
                            f"Distractor Context: {sample['distractor_passage']}\n"
                            f"Original Context: {sample['golden_passage']}"),
                    'labels': sample['distractor_answer']
                }
            
            elif sample_type == "cot":
                return {
                    'text': (f"Question: {sample['question']}\n"
                            f"Strategy: {sample['strategy']}\n"
                            f"Reasoning: {sample['reasoning']}\n"
                            f"Answer: {sample['answer']}\n"
                            f"References: {''.join(sample['references'])}"),
                    'labels': sample['answer']
                }
            
        except Exception as e:
            self.logger.warning(f"Failed to convert sample to training format: {e}")
            return None
    
    def _train_model(self, preparation_results: Dict) -> Dict:
        """Train the model on prepared data"""
        results = {
            'training_metrics': {},
            'model_path': None,
            'training_time': datetime.now().isoformat()
        }
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Add padding token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Tokenize datasets
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding=False,
                max_length=self.config.max_seq_length
            )
        
        train_dataset = preparation_results['training_dataset'].map(
            tokenize_function, batched=True
        )
        eval_dataset = preparation_results['validation_dataset'].map(
            tokenize_function, batched=True
        )
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=self.config.model_output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=True,
            dataloader_pin_memory=False
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # Start training
        self.logger.info("Starting model training...")
        training_result = self.trainer.train()
        
        # Save training results
        results['training_metrics'] = training_result.metrics
        results['model_path'] = f"{self.config.model_output_dir}/final"
        
        # Save the final model
        self.trainer.save_model(results['model_path'])
        self.tokenizer.save_pretrained(results['model_path'])
        
        self.logger.info(f"Model training completed. Saved to: {results['model_path']}")
        
        return results
    
    def _evaluate_model(self, training_results: Dict) -> Dict:
        """Evaluate the trained model"""
        results = {
            'evaluation_metrics': {},
            'evaluation_time': datetime.now().isoformat()
        }
        
        if not self.trainer or not hasattr(self.trainer, 'evaluate'):
            self.logger.warning("Trainer not available for evaluation")
            return results
        
        try:
            # Evaluate on validation set
            eval_metrics = self.trainer.evaluate()
            results['evaluation_metrics'] = eval_metrics
            
            # Additional evaluation can be added here
            # (e.g., on test sets, custom benchmarks, etc.)
            
            self.logger.info(f"Model evaluation completed. Loss: {eval_metrics.get('eval_loss', 'N/A')}")
            
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def _save_generated_data(self, generation_results: Dict) -> None:
        """Save generated data to files"""
        output_dir = Path(self.config.data_cache_dir)
        
        # Save seed samples
        with open(output_dir / "seed_samples.json", 'w', encoding='utf-8') as f:
            json.dump(generation_results['seed_samples'], f, ensure_ascii=False, indent=2)
        
        # Save distractor samples
        with open(output_dir / "distractor_samples.json", 'w', encoding='utf-8') as f:
            json.dump(generation_results['distractor_samples'], f, ensure_ascii=False, indent=2)
        
        # Save CoT samples
        with open(output_dir / "cot_samples.json", 'w', encoding='utf-8') as f:
            json.dump(generation_results['cot_samples'], f, ensure_ascii=False, indent=2)
    
    def _save_quality_reports(self, quality_results: Dict) -> None:
        """Save quality assessment reports"""
        output_dir = Path(self.config.output_dir) / "reports"
        
        with open(output_dir / "quality_reports.json", 'w', encoding='utf-8') as f:
            json.dump(quality_results, f, ensure_ascii=False, indent=2)
    
    def _save_datasets(self, preparation_results: Dict) -> None:
        """Save prepared datasets"""
        output_dir = Path(self.config.data_cache_dir)
        
        preparation_results['training_dataset'].to_json(output_dir / "train_dataset.json")
        preparation_results['validation_dataset'].to_json(output_dir / "val_dataset.json")
    
    def _save_pipeline_results(self, pipeline_results: Dict) -> None:
        """Save complete pipeline results"""
        output_dir = Path(self.config.output_dir)
        
        with open(output_dir / "pipeline_results.json", 'w', encoding='utf-8') as f:
            json.dump(pipeline_results, f, ensure_ascii=False, indent=2)
        
        # Generate summary report
        summary = {
            'timestamp': datetime.now().isoformat(),
            'success': pipeline_results['summary']['success'],
            'duration_seconds': pipeline_results['summary'].get('duration_seconds', 0),
            'total_samples': pipeline_results['summary'].get('total_samples', 0),
            'model_path': pipeline_results.get('training', {}).get('model_path'),
            'final_loss': pipeline_results.get('evaluation', {}).get('evaluation_metrics', {}).get('eval_loss')
        }
        
        with open(output_dir / "summary_report.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


# Example usage
if __name__ == "__main__":
    # Initialize and run the pipeline
    pipeline = TrainingPipeline()
    
    print("Starting PrismRAG training pipeline...")
    results = pipeline.run_pipeline()
    
    if results['summary']['success']:
        print("Pipeline completed successfully!")
        print(f"Total samples: {results['summary']['total_samples']}")
        print(f"Duration: {results['summary']['duration_seconds']:.1f} seconds")
        print(f"Model saved to: {results['training']['model_path']}")
    else:
        print("Pipeline failed!")
        print(f"Error: {results['summary']['error']}")