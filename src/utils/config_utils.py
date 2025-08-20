"""
Configuration Management Utilities for PrismRAG

This module provides configuration loading, validation, and management
functionality for the PrismRAG project.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
from functools import lru_cache

import logging
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration dataclass"""
    base_model: str = "meta-llama/Llama-3.1-70b-instruct"
    max_length: int = 4096
    temperature: float = 1.0
    top_p: float = 0.9
    device: str = "auto"
    torch_dtype: str = "float16"


@dataclass
class TrainingConfig:
    """Training configuration dataclass"""
    learning_rate: float = 1e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    save_total_limit: int = 3


@dataclass
class DataGenerationConfig:
    """Data generation configuration dataclass"""
    max_samples_per_source: int = 1000
    min_passage_length: int = 200
    max_passage_length: int = 1000
    difficulty_level: int = 8


@dataclass
class PathsConfig:
    """Paths configuration dataclass"""
    data_dir: str = "data"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    training_data_dir: str = "data/training"
    model_dir: str = "models"
    output_dir: str = "outputs"
    cache_dir: str = "cache"
    log_dir: str = "logs"


class ConfigManager:
    """
    Configuration manager for PrismRAG that handles loading, validating,
    and providing access to configuration settings.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file. If None, uses default path.
        """
        self.config_path = config_path or os.path.join("config", "default.yaml")
        self._config = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
        except FileNotFoundError:
            logger.warning(f"Configuration file {self.config_path} not found, using defaults")
            self._config = {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Configuration key in dot notation (e.g., 'model.base_model')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration as a dataclass"""
        return ModelConfig(
            base_model=self.get('model.base_model'),
            max_length=self.get('model.max_length'),
            temperature=self.get('model.temperature'),
            top_p=self.get('model.top_p'),
            device=self.get('model.device'),
            torch_dtype=self.get('model.torch_dtype')
        )
    
    def get_training_config(self) -> TrainingConfig:
        """Get training configuration as a dataclass"""
        return TrainingConfig(
            learning_rate=self.get('training.learning_rate'),
            batch_size=self.get('training.batch_size'),
            gradient_accumulation_steps=self.get('training.gradient_accumulation_steps'),
            num_epochs=self.get('training.num_epochs'),
            warmup_steps=self.get('training.warmup_steps'),
            weight_decay=self.get('training.weight_decay'),
            max_grad_norm=self.get('training.max_grad_norm'),
            save_steps=self.get('training.save_steps'),
            eval_steps=self.get('training.eval_steps'),
            logging_steps=self.get('training.logging_steps'),
            save_total_limit=self.get('training.save_total_limit')
        )
    
    def get_data_generation_config(self) -> DataGenerationConfig:
        """Get data generation configuration as a dataclass"""
        return DataGenerationConfig(
            max_samples_per_source=self.get('data_generation.seed_data.max_samples_per_source'),
            min_passage_length=self.get('data_generation.seed_data.min_passage_length'),
            max_passage_length=self.get('data_generation.seed_data.max_passage_length'),
            difficulty_level=self.get('data_generation.seed_data.difficulty_level')
        )
    
    def get_paths_config(self) -> PathsConfig:
        """Get paths configuration as a dataclass"""
        return PathsConfig(
            data_dir=self.get('paths.data_dir'),
            raw_data_dir=self.get('paths.raw_data_dir'),
            processed_data_dir=self.get('paths.processed_data_dir'),
            training_data_dir=self.get('paths.training_data_dir'),
            model_dir=self.get('paths.model_dir'),
            output_dir=self.get('paths.output_dir'),
            cache_dir=self.get('paths.cache_dir'),
            log_dir=self.get('paths.log_dir')
        )
    
    def validate_config(self) -> bool:
        """
        Validate the configuration for required settings.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        required_keys = [
            'model.base_model',
            'training.learning_rate',
            'paths.data_dir'
        ]
        
        missing_keys = []
        for key in required_keys:
            if self.get(key) is None:
                missing_keys.append(key)
        
        if missing_keys:
            logger.error(f"Missing required configuration keys: {missing_keys}")
            return False
        
        return True
    
    def ensure_directories(self) -> None:
        """Ensure all required directories exist"""
        paths_config = self.get_paths_config()
        
        directories = [
            paths_config.data_dir,
            paths_config.raw_data_dir,
            paths_config.processed_data_dir,
            paths_config.training_data_dir,
            paths_config.model_dir,
            paths_config.output_dir,
            paths_config.cache_dir,
            paths_config.log_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")


# Global configuration instance
@lru_cache(maxsize=1)
def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """
    Get a cached instance of the configuration manager.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ConfigManager instance
    """
    return ConfigManager(config_path)


# Example usage
if __name__ == "__main__":
    # Initialize configuration
    config_manager = ConfigManager()
    
    # Validate configuration
    if config_manager.validate_config():
        print("Configuration is valid")
        
        # Ensure directories exist
        config_manager.ensure_directories()
        
        # Access configuration values
        model_config = config_manager.get_model_config()
        print(f"Model: {model_config.base_model}")
        
        training_config = config_manager.get_training_config()
        print(f"Learning rate: {training_config.learning_rate}")
    else:
        print("Configuration validation failed")