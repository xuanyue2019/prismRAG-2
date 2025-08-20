# PrismRAG Configuration Guide

Complete guide to configuring and customizing the PrismRAG system for optimal performance.

## Overview

PrismRAG uses a hierarchical YAML configuration system that allows fine-grained control over all aspects of the system, from data generation to model training and evaluation.

## Configuration Structure

### Main Configuration File
The primary configuration is stored in [`config/default.yaml`](../config/default.yaml) and follows this structure:

```yaml
# PrismRAG Configuration File

# Model settings
model:
  base_model: "meta-llama/Llama-3.1-70b-instruct"
  max_length: 4096
  temperature: 1.0
  top_p: 0.9
  device: "auto"
  torch_dtype: "float16"

# Training settings
training:
  learning_rate: 1e-5
  batch_size: 4
  gradient_accumulation_steps: 8
  num_epochs: 3
  # ... more training parameters

# Data generation settings
data_generation:
  seed_data:
    max_samples_per_source: 1000
    min_passage_length: 200
    max_passage_length: 1000
    difficulty_level: 8
  
  distractor:
    max_iterations: 5
    quality_threshold: 4
    batch_size: 16
    relevance_weight: 0.4      # Configurable evaluation weights
    distraction_weight: 0.4    # Configurable evaluation weights  
    format_weight: 0.2         # Configurable evaluation weights
  
  strategic_cot:
    max_iterations: 10
    random_attempts: 6
    quality_threshold: 4
    batch_size: 8
    strategy_depth: 3

# Data sources configuration
data_sources:
  wikipedia:
    enabled: true
    num_pages: 1000
    min_words: 500
    max_words: 7000
    # ... more Wikipedia settings
  
  web_search:
    enabled: true
    max_pages_per_query: 10
    max_words_per_page: 3000
    search_engines: ["google", "bing"]
    # ... more web search settings

# Evaluation settings
evaluation:
  benchmarks:
    - "crag"
    - "covidqa"
    - "delucionqa"
    - "emanual"
    - "expertqa"
    - "finqa"
    - "hagrid"
    - "hotpotqa"
    - "ms_marco"
    - "pubmedqa"
    - "tatqa"
    - "techqa"
  
  metrics:
    - "factuality_score"
    - "accuracy"
    - "hallucination_rate"
    - "missing_rate"
    - "completeness"
    - "relevance"
  
  llm_as_judge:
    enabled: true
    model: "gpt-4"
    temperature: 0.3
    max_tokens: 500

# Paths configuration
paths:
  data_dir: "data"
  raw_data_dir: "data/raw"
  processed_data_dir: "data/processed"
  training_data_dir: "data/training"
  model_dir: "models"
  output_dir: "outputs"
  cache_dir: "cache"
  log_dir: "logs"

# API keys (environment variables)
api_keys:
  openai: ${OPENAI_API_KEY}
  google_search: ${GOOGLE_SEARCH_API_KEY}
  serpapi: ${SERPAPI_API_KEY}
  huggingface: ${HUGGINGFACE_HUB_TOKEN}

# Logging configuration
logging:
  level: "INFO"
  use_wandb: true
  wandb_project: "prismrag"
  wandb_entity: null
  log_file: "prismrag.log"
  console_output: true

# Performance optimization
performance:
  use_gradient_checkpointing: true
  use_flash_attention: true
  mixed_precision: "fp16"
  dataloader_num_workers: 4
  prefetch_factor: 2
  pin_memory: true

# Experimental features
experimental:
  use_raft: false
  use_star: false
  use_llm_quoter: false
  enable_multilingual: false
  enable_domain_specific: false

# Version info
version: "1.0.0"
last_updated: "2025-01-19"
```

## Key Configuration Sections

### Model Configuration
```yaml
model:
  base_model: "meta-llama/Llama-3.1-70b-instruct"  # Base model for generation and training
  max_length: 4096                                  # Maximum sequence length
  temperature: 1.0                                  # Sampling temperature
  top_p: 0.9                                        # Nucleus sampling parameter
  device: "auto"                                    # Device allocation
  torch_dtype: "float16"                            # Precision
```

### Training Configuration
```yaml
training:
  learning_rate: 1e-5                               # Learning rate
  batch_size: 4                                     # Batch size
  gradient_accumulation_steps: 8                    # Gradient accumulation
  num_epochs: 3                                     # Training epochs
  warmup_steps: 100                                 # Warmup steps
  weight_decay: 0.01                                # Weight decay
  max_grad_norm: 1.0                                # Gradient clipping
  save_steps: 500                                   # Checkpoint saving frequency
  eval_steps: 500                                   # Evaluation frequency
```

### Data Generation Configuration

#### Seed Data Generation
```yaml
seed_data:
  max_samples_per_source: 1000                      # Max samples per data source
  min_passage_length: 200                           # Minimum passage length
  max_passage_length: 1000                          # Maximum passage length
  difficulty_level: 8                               # Question difficulty (1-10)
```

#### Distractor Generation
```yaml
distractor:
  max_iterations: 5                                 # Max generation attempts
  quality_threshold: 4                              # Minimum quality score (1-5)
  batch_size: 16                                    # Batch size for generation
  relevance_weight: 0.4                             # Weight for relevance score
  distraction_weight: 0.4                           # Weight for distraction score
  format_weight: 0.2                                # Weight for format score
```

#### Strategic CoT Generation
```yaml
strategic_cot:
  max_iterations: 10                                # Max generation attempts
  random_attempts: 6                                # Random attempts before critique
  quality_threshold: 4                              # Minimum quality score (1-4)
  batch_size: 8                                     # Batch size for generation
  strategy_depth: 3                                 # Depth of reasoning strategies
```

### Data Sources Configuration

#### Wikipedia Configuration
```yaml
wikipedia:
  enabled: true                                     # Enable Wikipedia data source
  num_pages: 1000                                   # Number of pages to fetch
  min_words: 500                                    # Minimum article length
  max_words: 7000                                   # Maximum article length
  min_lines: 10                                     # Minimum lines
  max_lines: 1000                                   # Maximum lines
  chunk_size_min: 250                               # Minimum chunk size
  chunk_size_max: 1000                              # Maximum chunk size
  languages: ["en"]                                 # Supported languages
  categories: ["science", "history", "technology", "culture"]  # Content categories
```

#### Web Search Configuration
```yaml
web_search:
  enabled: true                                     # Enable web search data source
  max_pages_per_query: 10                           # Max pages per search query
  max_words_per_page: 3000                          # Max words per page
  search_engines: ["google", "bing"]                # Supported search engines
  query_categories: ["news", "academic", "general_knowledge"]  # Query categories
  time_sensitive: true                              # Include time-sensitive results
```

### Evaluation Configuration

#### Benchmarks
```yaml
benchmarks:
  - "crag"                                          # CRAG benchmark
  - "covidqa"                                       # COVID-QA benchmark
  - "delucionqa"                                    # DelucionQA benchmark
  - "emanual"                                       # EManual benchmark
  - "expertqa"                                      # ExpertQA benchmark
  - "finqa"                                         # FinQA benchmark
  - "hagrid"                                        # HAGRID benchmark
  - "hotpotqa"                                      # HotpotQA benchmark
  - "ms_marco"                                      # MS MARCO benchmark
  - "pubmedqa"                                      # PubMedQA benchmark
  - "tatqa"                                         # TAT-QA benchmark
  - "techqa"                                        # TechQA benchmark
```

#### Evaluation Metrics
```yaml
metrics:
  - "factuality_score"                              # Factuality assessment
  - "accuracy"                                      # Answer accuracy
  - "hallucination_rate"                            # Hallucination detection
  - "missing_rate"                                  # Missing information rate
  - "completeness"                                  # Answer completeness
  - "relevance"                                     # Relevance to question
```

#### LLM-as-Judge Configuration
```yaml
llm_as_judge:
  enabled: true                                     # Enable LLM evaluation
  model: "gpt-4"                                    # Evaluation model
  temperature: 0.3                                  # Evaluation temperature
  max_tokens: 500                                   # Max tokens per evaluation
```

### Paths Configuration
```yaml
paths:
  data_dir: "data"                                  # Root data directory
  raw_data_dir: "data/raw"                          # Raw data storage
  processed_data_dir: "data/processed"              # Processed data storage
  training_data_dir: "data/training"                # Training data storage
  model_dir: "models"                               # Model storage
  output_dir: "outputs"                             # Output directory
  cache_dir: "cache"                                # Cache directory
  log_dir: "logs"                                   # Log directory
```

### API Keys Configuration
```yaml
api_keys:
  openai: ${OPENAI_API_KEY}                         # OpenAI API key
  google_search: ${GOOGLE_SEARCH_API_KEY}           # Google Search API key
  serpapi: ${SERPAPI_API_KEY}                       # SerpAPI key
  huggingface: ${HUGGINGFACE_HUB_TOKEN}             # Hugging Face token
```

### Logging Configuration
```yaml
logging:
  level: "INFO"                                     # Logging level
  use_wandb: true                                   # Enable Weights & Biases
  wandb_project: "prismrag"                         # W&B project name
  wandb_entity: null                                # W&B entity (username)
  log_file: "prismrag.log"                          # Log file name
  console_output: true                              # Console output
```

### Performance Configuration
```yaml
performance:
  use_gradient_checkpointing: true                  # Gradient checkpointing
  use_flash_attention: true                         # Flash attention optimization
  mixed_precision: "fp16"                           # Mixed precision training
  dataloader_num_workers: 4                         # Data loader workers
  prefetch_factor: 2                                # Prefetch factor
  pin_memory: true                                  # Pin memory for faster transfer
```

## Environment Variables

PrismRAG uses environment variables for sensitive configuration:

```bash
# Required API keys
export OPENAI_API_KEY="your_openai_api_key"
export GOOGLE_SEARCH_API_KEY="your_google_search_key"
export SERPAPI_API_KEY="your_serpapi_key"
export HUGGINGFACE_HUB_TOKEN="your_hf_token"

# Optional configuration overrides
export PRISM_RAG_MODEL="your/custom-model"
export PRISM_RAG_DEVICE="cuda:0"
export PRISM_RAG_DATA_DIR="/path/to/custom/data"
```

## Configuration Management

### Using ConfigManager
The [`ConfigManager`](../src/utils/config_utils.py) class provides unified access to configuration:

```python
from src.utils import ConfigManager

# Initialize with default configuration
config = ConfigManager()

# Access configuration values
model_name = config.get('model.base_model')
learning_rate = config.get('training.learning_rate')

# Override configuration
config.set('model.base_model', 'custom/model-name')
config.set('training.batch_size', 8)

# Save configuration changes
config.save('config/custom_config.yaml')
```

### Creating Custom Configurations

1. **Copy default configuration:**
```bash
cp config/default.yaml config/custom_config.yaml
```

2. **Modify desired settings:**
```yaml
# config/custom_config.yaml
model:
  base_model: "your/custom-model"
  max_length: 2048

training:
  learning_rate: 2e-5
  batch_size: 8
```

3. **Use custom configuration:**
```python
from src.utils import ConfigManager

config = ConfigManager('config/custom_config.yaml')
```

### Configuration Validation

The system includes configuration validation:

```python
# Validate configuration
if config.validate():
    print("Configuration is valid")
else:
    errors = config.get_validation_errors()
    print(f"Configuration errors: {errors}")
```

## Best Practices

### 1. Model Selection
- Use larger models (70B+) for better generation quality
- Consider domain-specific models for specialized tasks
- Balance model size with available hardware

### 2. Quality Thresholds
- Start with moderate thresholds (3.0-3.5) and adjust based on needs
- Higher thresholds produce better quality but fewer samples
- Monitor acceptance rates and adjust thresholds accordingly

### 3. Performance Optimization
- Use mixed precision (fp16) for faster training
- Enable gradient checkpointing for memory efficiency
- Adjust batch size based on available GPU memory

### 4. Data Source Configuration
- Enable multiple data sources for diversity
- Adjust chunk sizes based on content type
- Use appropriate categories for your domain

### 5. Evaluation Setup
- Include relevant benchmarks for your use case
- Use multiple evaluation metrics for comprehensive assessment
- Enable LLM-as-judge for qualitative evaluation

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure environment variables are set
   - Check API key permissions and quotas

2. **Memory Issues**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use smaller model variants

3. **Quality Issues**
   - Adjust quality thresholds
   - Increase generation iterations
   - Review data source configuration

4. **Performance Issues**
   - Enable performance optimizations
   - Use appropriate hardware
   - Monitor system resources

## Advanced Configuration

### Custom Evaluation Weights
Modify distractor evaluation weights in the configuration:

```yaml
data_generation:
  distractor:
    relevance_weight: 0.5    # Increase relevance importance
    distraction_weight: 0.3  # Decrease distraction importance  
    format_weight: 0.2       # Keep format importance
```

### Domain-Specific Configuration
Create domain-specific configurations:

```yaml
# config/medical_config.yaml
data_sources:
  wikipedia:
    categories: ["medicine", "biology", "health"]
  
  web_search:
    query_categories: ["medical_research", "clinical_studies"]

experimental:
  enable_domain_specific: true
```

### Multi-Lingual Support
Enable multi-lingual processing:

```yaml
data_sources:
  wikipedia:
    languages: ["en", "zh", "es", "fr", "de"]
  
experimental:
  enable_multilingual: true
```

## Version Compatibility

Ensure configuration compatibility with different versions:

- **v1.0.0**: Initial release with current configuration structure
- Check [`CHANGELOG.md`](../CHANGELOG.md) for version-specific changes

## Support

For configuration-related issues:
1. Check the [FAQ](../docs/FAQ.md)
2. Review [API documentation](../docs/API.md)
3. Create an issue on GitHub
4. Contact the development team

---

*Last updated: 2025-01-19*
*Configuration version: 1.0.0*