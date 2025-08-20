# PrismRAG API Documentation

Complete API reference for the PrismRAG system, covering all modules, classes, and methods.

## Overview

PrismRAG provides a comprehensive API for data acquisition, generation, training, and evaluation. This document covers all public interfaces and their usage.

## Core Modules

### 1. Configuration Management ([`src/utils/config_utils.py`](../src/utils/config_utils.py))

#### ConfigManager Class
```python
class ConfigManager:
    """Unified configuration management for PrismRAG"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to custom configuration file
        """
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation key"""
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
    
    def save(self, filepath: str) -> None:
        """Save configuration to file"""
    
    def validate(self) -> bool:
        """Validate configuration structure"""
    
    def get_validation_errors(self) -> List[str]:
        """Get configuration validation errors"""
```

**Usage Example:**
```python
from src.utils import ConfigManager

# Initialize with default config
config = ConfigManager()

# Access configuration
model_name = config.get('model.base_model')
learning_rate = config.get('training.learning_rate')

# Modify configuration
config.set('model.base_model', 'custom/model')
config.set('training.batch_size', 8)

# Save custom configuration
config.save('config/custom_config.yaml')
```

### 2. Data Acquisition ([`src/data_acquisition/`](../src/data_acquisition/))

#### WikipediaClient Class
```python
class WikipediaClient:
    """Client for fetching and processing Wikipedia data"""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """Initialize Wikipedia client"""
    
    def search_and_extract(self, query: str, max_articles: int = 5) -> List[Dict]:
        """
        Search Wikipedia and extract articles.
        
        Args:
            query: Search query
            max_articles: Maximum articles to return
            
        Returns:
            List of article dictionaries with content and metadata
        """
    
    def get_article(self, title: str) -> Optional[Dict]:
        """Get specific Wikipedia article by title"""
```

**Usage Example:**
```python
from src.data_acquisition import WikipediaClient

client = WikipediaClient()
articles = client.search_and_extract("machine learning", max_articles=3)

for article in articles:
    print(f"Title: {article['title']}")
    print(f"Content: {article['content'][:200]}...")
```

#### WebSearchClient Class
```python
class WebSearchClient:
    """Client for web search data acquisition"""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """Initialize web search client"""
    
    def search(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Perform web search and extract content.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            
        Returns:
            List of search results with content and metadata
        """
```

**Usage Example:**
```python
from src.data_acquisition import WebSearchClient

client = WebSearchClient()
results = client.search("recent AI developments", max_results=5)

for result in results:
    print(f"Title: {result['title']}")
    print(f"URL: {result['url']}")
    print(f"Content: {result['content'][:200]}...")
```

### 3. Data Generation ([`src/data_generation/`](../src/data_generation/))

#### SeedDataGenerator Class
```python
class SeedDataGenerator:
    """Generates seed QA pairs from documents"""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """Initialize seed data generator"""
    
    def generate_from_document(self, document: str, num_questions: int = 5) -> List[Dict]:
        """
        Generate QA pairs from a single document.
        
        Args:
            document: Source document text
            num_questions: Number of questions to generate
            
        Returns:
            List of QA samples
        """
    
    def generate_batch(self, documents: List[str], num_samples: int = 100, 
                      max_attempts: int = 3) -> List[Dict]:
        """
        Generate QA pairs from multiple documents.
        
        Args:
            documents: List of source documents
            num_samples: Total samples to generate
            max_attempts: Maximum generation attempts per document
            
        Returns:
            List of QA samples
        """
```

**Usage Example:**
```python
from src.data_generation import SeedDataGenerator

generator = SeedDataGenerator()
documents = ["Long document text 1...", "Long document text 2..."]
samples = generator.generate_batch(documents, num_samples=50)

for sample in samples[:3]:
    print(f"Q: {sample['question']}")
    print(f"A: {sample['answer']}")
    print(f"Passage: {sample['passage'][:100]}...")
```

#### DistractorGenerator Class
```python
class DistractorGenerator:
    """Generates distractor passages and questions"""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """Initialize distractor generator"""
    
    def generate_distractor(self, question: str, answer: str, passage: str,
                          user_time: Optional[str] = None, 
                          location: Optional[str] = None) -> Optional[Dict]:
        """
        Generate distractor for a QA pair.
        
        Args:
            question: Original question
            answer: Original answer
            passage: Original passage
            user_time: User time context
            location: User location context
            
        Returns:
            Distractor sample dictionary
        """
    
    def generate_batch(self, samples: List[Dict], max_workers: int = 4) -> List[Dict]:
        """
        Generate distractors for multiple samples.
        
        Args:
            samples: List of seed samples
            max_workers: Maximum parallel workers
            
        Returns:
            List of distractor samples
        """
```

**Usage Example:**
```python
from src.data_generation import DistractorGenerator

generator = DistractorGenerator()
seed_sample = {
    "question": "What is the capital of France?",
    "answer": "Paris",
    "passage": "France is a country in Europe. Paris is its capital..."
}

distractor = generator.generate_distractor(**seed_sample)

if distractor:
    print(f"Original Q: {distractor['question']}")
    print(f"Distractor Q: {distractor['open_ended_question']}")
    print(f"Distractor A: {distractor['distractor_answer']}")
```

#### StrategicCoTGenerator Class
```python
class StrategicCoTGenerator:
    """Generates strategic Chain-of-Thought reasoning"""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """Initialize CoT generator"""
    
    def generate_strategic_cot(self, question: str, references: List[str],
                             ground_truth_answer: str,
                             user_context: Optional[str] = None) -> Optional[Dict]:
        """
        Generate strategic CoT for a question.
        
        Args:
            question: The question to answer
            references: Reference documents
            ground_truth_answer: Correct answer for evaluation
            user_context: Additional user context
            
        Returns:
            CoT sample dictionary
        """
    
    def generate_batch(self, samples: List[Tuple[str, List[str], str]],
                     user_context: Optional[str] = None) -> List[Dict]:
        """
        Generate CoT for multiple samples.
        
        Args:
            samples: List of (question, references, ground_truth) tuples
            user_context: Additional user context
            
        Returns:
            List of CoT samples
        """
```

**Usage Example:**
```python
from src.data_generation import StrategicCoTGenerator

generator = StrategicCoTGenerator()
sample = (
    "What is the capital of France?",
    ["France is a country in Europe. Paris is its capital city..."],
    "Paris"
)

cot_sample = generator.generate_strategic_cot(*sample)

if cot_sample:
    print(f"Strategy: {cot_sample['strategy'][:100]}...")
    print(f"Reasoning: {cot_sample['reasoning'][:100]}...")
    print(f"Answer: {cot_sample['answer']}")
```

### 4. Quality Assessment ([`src/data_generation/`](../src/data_generation/))

#### DataQualityAssessor Class
```python
class DataQualityAssessor:
    """Comprehensive data quality assessment"""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """Initialize quality assessor"""
    
    def assess_distractor_sample(self, question: str, answer: str, golden_passage: str,
                               distractor_passage: str, open_ended_question: str,
                               distractor_answer: str, user_time: Optional[str] = None,
                               location: Optional[str] = None,
                               max_attempts: int = 3) -> Tuple[Optional[Dict], Dict]:
        """
        Assess quality of distractor sample.
        
        Returns:
            Tuple of (evaluation_dict, quality_metrics)
        """
    
    def assess_cot_sample(self, question: str, references: List[str], strategy: str,
                        reasoning: str, answer: str, ground_truth: str,
                        max_attempts: int = 3) -> Tuple[Optional[int], Optional[int], Dict]:
        """
        Assess quality of CoT sample.
        
        Returns:
            Tuple of (reasoning_score, answer_score, quality_metrics)
        """
    
    def assess_batch(self, samples: List[Dict], sample_type: str = "distractor",
                   max_workers: int = 4) -> List[Tuple[Dict, Dict]]:
        """
        Assess batch of samples.
        
        Returns:
            List of (sample, quality_metrics) tuples
        """
```

**Usage Example:**
```python
from src.data_generation import DataQualityAssessor

assessor = DataQualityAssessor()
sample = {
    "question": "What is the capital of France?",
    "answer": "Paris",
    "golden_passage": "France is in Europe...",
    "distractor_passage": "France's main city is Lyon...",
    "open_ended_question": "Which city is capital of France?",
    "distractor_answer": "Lyon"
}

evaluation, metrics = assessor.assess_distractor_sample(**sample)

print(f"Relevance score: {metrics.relevance_score}")
print(f"Distraction score: {metrics.distraction_score}")
print(f"Overall score: {metrics.overall_score}")
```

#### DataValidator Class
```python
class DataValidator:
    """Data validation against schema and quality rules"""
    
    def __init__(self):
        """Initialize data validator"""
    
    def validate_distractor_sample(self, sample: Dict) -> Dict:
        """Validate distractor sample"""
    
    def validate_cot_sample(self, sample: Dict) -> Dict:
        """Validate CoT sample"""
    
    def validate_seed_qa_sample(self, sample: Dict) -> Dict:
        """Validate seed QA sample"""
    
    def validate_batch(self, samples: List[Dict], sample_type: str = "distractor") -> List[Tuple[Dict, Dict]]:
        """Validate batch of samples"""
```

**Usage Example:**
```python
from src.data_generation import DataValidator

validator = DataValidator()
sample = {
    "question": "What is the capital of France?",
    "answer": "Paris",
    "golden_passage": "France is in Europe...",
    # ... other fields
}

result = validator.validate_distractor_sample(sample)

print(f"Valid: {result.is_valid}")
print(f"Errors: {result.errors}")
print(f"Warnings: {result.warnings}")
```

#### QualityMonitor Class
```python
class QualityMonitor:
    """Real-time quality monitoring and feedback"""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """Initialize quality monitor"""
    
    def track_sample_quality(self, sample: Dict, sample_type: str,
                           metrics: Optional[Dict] = None,
                           validation_result: Optional[Dict] = None) -> None:
        """Track quality of a single sample"""
    
    def track_batch_quality(self, samples: List[Dict], sample_type: str,
                          batch_id: Optional[str] = None) -> Dict:
        """Track quality of a batch of samples"""
    
    def get_quality_feedback(self) -> List[str]:
        """Get actionable quality feedback"""
    
    def generate_quality_report(self) -> Dict:
        """Generate comprehensive quality report"""
```

**Usage Example:**
```python
from src.data_generation import QualityMonitor

monitor = QualityMonitor()

# Track individual sample
monitor.track_sample_quality(sample, "distractor", metrics, validation_result)

# Track batch
batch_summary = monitor.track_batch_quality(samples, "distractor", "batch_001")

# Get feedback
feedback = monitor.get_quality_feedback()
for item in feedback:
    print(f"Feedback: {item}")
```

### 5. Training Pipeline ([`src/training/`](../src/training/))

#### TrainingPipeline Class
```python
class TrainingPipeline:
    """End-to-end training pipeline"""
    
    def __init__(self, config: Optional[TrainingPipelineConfig] = None):
        """Initialize training pipeline"""
    
    def run_pipeline(self) -> Dict:
        """
        Run complete training pipeline.
        
        Returns:
            Pipeline execution results
        """
    
    def run_phase(self, phase_name: str, *args, **kwargs) -> Dict:
        """Run specific pipeline phase"""
```

**Usage Example:**
```python
from src.training import TrainingPipeline

pipeline = TrainingPipeline()
results = pipeline.run_pipeline()

if results['summary']['success']:
    print("Pipeline completed successfully!")
    print(f"Model saved to: {results['training']['model_path']}")
else:
    print(f"Pipeline failed: {results['summary']['error']}")
```

### 6. Evaluation ([`src/evaluation/`](../src/evaluation/))

#### BenchmarkEvaluator Class
```python
class BenchmarkEvaluator:
    """Multi-benchmark model evaluation"""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """Initialize benchmark evaluator"""
    
    def evaluate_all_benchmarks(self) -> Dict:
        """Evaluate on all supported benchmarks"""
    
    def evaluate_benchmark(self, benchmark_name: str) -> Dict:
        """Evaluate on specific benchmark"""
    
    def generate_evaluation_report(self) -> str:
        """Generate comprehensive evaluation report"""
```

**Usage Example:**
```python
from src.evaluation import BenchmarkEvaluator

evaluator = BenchmarkEvaluator(model_path="./models/final")
summary = evaluator.evaluate_all_benchmarks()

print(f"Overall score: {summary.overall_score:.3f}")
print("Benchmark results:")
for benchmark, result in summary.benchmark_results.items():
    print(f"  {benchmark}: {np.mean(list(result.metrics.values())):.3f}")
```

## Utility Modules

### Logging Utilities ([`src/utils/logging_utils.py`](../src/utils/logging_utils.py))
```python
def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Setup standardized logging configuration"""

def get_logger(name: str) -> logging.Logger:
    """Get configured logger instance"""
```

### Data Utilities ([`src/utils/data_utils.py`](../src/utils/data_utils.py))
```python
def load_jsonl(filepath: str) -> List[Dict]:
    """Load JSONL file"""

def save_jsonl(data: List[Dict], filepath: str) -> None:
    """Save data to JSONL file"""

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Chunk text into overlapping segments"""
```

### Model Utilities ([`src/utils/model_utils.py`](../src/utils/model_utils.py))
```python
def load_model(model_name: str, device: str = "auto") -> Tuple[Any, Any]:
    """Load model and tokenizer"""

def generate_text(prompt: str, model: Any, tokenizer: Any, 
                 max_length: int = 512, temperature: float = 0.7) -> str:
    """Generate text using model"""
```

## Advanced Usage

### Custom Configuration
```python
from src.utils import ConfigManager

# Create custom configuration
config = ConfigManager()
config.set('model.base_model', 'your/custom-model')
config.set('training.learning_rate', 2e-5)
config.set('data_generation.distractor.relevance_weight', 0.5)

# Use custom config with components
from src.data_generation import DistractorGenerator
generator = DistractorGenerator(config=config)
```

### Batch Processing
```python
from src.data_generation import SeedDataGenerator, DistractorGenerator
from src.data_acquisition import WikipediaClient

# Acquire data
client = WikipediaClient()
articles = client.search_and_extract("artificial intelligence", 10)

# Generate seed data
seed_generator = SeedDataGenerator()
seed_samples = seed_generator.generate_batch(
    [article['content'] for article in articles],
    num_samples=100
)

# Generate distractors
distractor_generator = DistractorGenerator()
distractor_samples = distractor_generator.generate_batch(
    seed_samples[:50],  # First 50 samples
    max_workers=8
)
```

### Quality-Driven Generation
```python
from src.data_generation import StrategicCoTGenerator, DataQualityAssessor

generator = StrategicCoTGenerator()
assessor = DataQualityAssessor()

# Generate with quality assurance
high_quality_samples = []
for attempt in range(5):  # Multiple attempts
    cot_sample = generator.generate_strategic_cot(question, references, answer)
    if cot_sample:
        _, _, metrics = assessor.assess_cot_sample(
            question, references,
            cot_sample['strategy'],
            cot_sample['reasoning'],
            cot_sample['answer'],
            answer
        )
        if metrics.overall_score >= 3.5:  # Quality threshold
            high_quality_samples.append(cot_sample)
            break
```

## Error Handling

All API methods include comprehensive error handling:

```python
try:
    from src.data_acquisition import WikipediaClient
    client = WikipediaClient()
    articles = client.search_and_extract("machine learning", 5)
except Exception as e:
    print(f"Error acquiring data: {e}")
    # Fallback to alternative data source or cached data
```

## Performance Considerations

### Memory Management
```python
# Use smaller batch sizes for memory-constrained environments
config.set('training.batch_size', 2)
config.set('data_generation.distractor.batch_size', 8)

# Enable memory optimizations
config.set('performance.use_gradient_checkpointing', True)
config.set('performance.use_flash_attention', True)
```

### Parallel Processing
```python
# Use multiple workers for data generation
distractor_samples = distractor_generator.generate_batch(
    samples,
    max_workers=8  # Adjust based on CPU cores
)

# Batch processing for efficiency
batch_size = 32
for i in range(0, len(data), batch_size):
    batch = data[i:i + batch_size]
    process_batch(batch)
```

## Best Practices

### 1. Configuration Management
```python
# Use environment-specific configurations
import os
env = os.getenv('ENVIRONMENT', 'development')
config_file = f'config/{env}_config.yaml'

if os.path.exists(config_file):
    config = ConfigManager(config_file)
else:
    config = ConfigManager()
```

### 2. Error Recovery
```python
# Implement retry logic with exponential backoff
import time
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_with_retry(generator, *args, **kwargs):
    return generator.generate(*args, **kwargs)
```

### 3. Resource Cleanup
```python
# Use context managers for resource cleanup
from contextlib import contextmanager

@contextmanager
def managed_generator(generator_class, *args, **kwargs):
    generator = generator_class(*args, **kwargs)
    try:
        yield generator
    finally:
        # Cleanup resources if needed
        pass

with managed_generator(DistractorGenerator) as generator:
    results = generator.generate_batch(samples)
```

## Version Compatibility

- **v1.0.0**: All APIs are stable and backward compatible
- Check [`CHANGELOG.md`](../CHANGELOG.md) for API changes between versions

## Support

For API-related issues:
1. Check this documentation
2. Review example usage in [`examples/`](../examples/)
3. Examine source code for detailed implementation
4. Create GitHub issues for bugs or feature requests

---

*Last updated: 2025-01-19*
*API version: 1.0.0*