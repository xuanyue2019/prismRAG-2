"""
Logging Utilities for PrismRAG

This module provides comprehensive logging functionality with support for
console logging, file logging, and integration with Weights & Biases.
"""

import os
import sys
import logging
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime

import loguru
from loguru import logger

# Remove default logger
logger.remove()

class LoggingManager:
    """
    Advanced logging manager for PrismRAG that supports multiple outputs
    and integration with external services like Weights & Biases.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the logging manager.
        
        Args:
            config: Configuration dictionary with logging settings
        """
        self.config = config or {}
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Setup logging configuration based on provided config"""
        log_level = self.config.get('level', 'INFO')
        use_wandb = self.config.get('use_wandb', False)
        log_file = self.config.get('log_file', 'prismrag.log')
        console_output = self.config.get('console_output', True)
        
        # Create log directory if it doesn't exist
        log_dir = self.config.get('log_dir', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure loguru logger
        log_file_path = os.path.join(log_dir, log_file)
        
        # Format for log messages
        format_str = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        
        # Add file handler
        logger.add(
            log_file_path,
            level=log_level,
            format=format_str,
            rotation="10 MB",  # Rotate when file reaches 10MB
            retention="30 days",  # Keep logs for 30 days
            compression="zip",  # Compress rotated files
            enqueue=True,  # Asynchronous logging
            backtrace=True,  # Enable traceback
            diagnose=True  # Enable variable values in traceback
        )
        
        # Add console handler if enabled
        if console_output:
            logger.add(
                sys.stderr,
                level=log_level,
                format=format_str,
                colorize=True,  # Enable colors in console
                enqueue=True
            )
        
        # Initialize W&B if enabled
        if use_wandb:
            self._init_wandb()
    
    def _init_wandb(self) -> None:
        """Initialize Weights & Biases integration"""
        try:
            import wandb
            
            wandb_project = self.config.get('wandb_project', 'prismrag')
            wandb_entity = self.config.get('wandb_entity')
            
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                config=self.config
            )
            
            # Add wandb handler
            class WandbHandler:
                def write(self, message):
                    # Extract log level and message from loguru format
                    if "|" in message:
                        parts = message.split("|")
                        if len(parts) >= 4:
                            level = parts[1].strip().lower()
                            log_message = parts[3].strip()
                            
                            # Log to wandb
                            wandb.log({"log": log_message, "level": level})
                
                def flush(self):
                    pass
            
            logger.add(WandbHandler(), level=self.config.get('level', 'INFO'))
            
        except ImportError:
            logger.warning("Weights & Biases not installed. Skipping W&B integration.")
        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}")
    
    def get_logger(self, name: Optional[str] = None) -> loguru.Logger:
        """
        Get a logger instance with the specified name.
        
        Args:
            name: Name for the logger (usually __name__)
            
        Returns:
            Configured logger instance
        """
        if name:
            return logger.bind(name=name)
        return logger
    
    def log_experiment_start(self, experiment_name: str, params: Dict[str, Any]) -> None:
        """
        Log the start of an experiment.
        
        Args:
            experiment_name: Name of the experiment
            params: Experiment parameters
        """
        logger.info(f"🚀 Starting experiment: {experiment_name}")
        logger.info(f"📋 Parameters: {params}")
        
        if 'wandb' in sys.modules:
            import wandb
            wandb.run.name = experiment_name if wandb.run.name is None else wandb.run.name
    
    def log_experiment_end(self, experiment_name: str, results: Dict[str, Any]) -> None:
        """
        Log the end of an experiment with results.
        
        Args:
            experiment_name: Name of the experiment
            results: Experiment results
        """
        logger.info(f"✅ Experiment completed: {experiment_name}")
        logger.info(f"📊 Results: {results}")
        
        if 'wandb' in sys.modules:
            import wandb
            for key, value in results.items():
                wandb.log({key: value})
    
    def log_data_generation(self, stage: str, count: int, details: Dict[str, Any]) -> None:
        """
        Log data generation progress.
        
        Args:
            stage: Data generation stage
            count: Number of items generated
            details: Additional details
        """
        logger.info(f"📦 {stage}: Generated {count} items")
        logger.debug(f"Details: {details}")
    
    def log_training_progress(self, epoch: int, step: int, metrics: Dict[str, float]) -> None:
        """
        Log training progress.
        
        Args:
            epoch: Current epoch
            step: Current step
            metrics: Training metrics
        """
        logger.info(f"🎯 Epoch {epoch}, Step {step}: {metrics}")
        
        if 'wandb' in sys.modules:
            import wandb
            wandb.log({"epoch": epoch, "step": step, **metrics})
    
    def log_evaluation_results(self, benchmark: str, metrics: Dict[str, float]) -> None:
        """
        Log evaluation results.
        
        Args:
            benchmark: Benchmark name
            metrics: Evaluation metrics
        """
        logger.info(f"📈 {benchmark} results: {metrics}")
        
        if 'wandb' in sys.modules:
            import wandb
            for metric_name, value in metrics.items():
                wandb.log({f"{benchmark}/{metric_name}": value})
    
    def catch_exceptions(self, func):
        """
        Decorator to catch and log exceptions.
        
        Args:
            func: Function to decorate
            
        Returns:
            Wrapped function
        """
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(f"Exception in {func.__name__}: {e}")
                raise
        return wrapper


# Global logging manager instance
_logging_manager = None

def setup_logging(config: Optional[Dict[str, Any]] = None) -> LoggingManager:
    """
    Setup and return the global logging manager.
    
    Args:
        config: Logging configuration
        
    Returns:
        LoggingManager instance
    """
    global _logging_manager
    if _logging_manager is None:
        _logging_manager = LoggingManager(config)
    return _logging_manager

def get_logger(name: Optional[str] = None) -> loguru.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    global _logging_manager
    if _logging_manager is None:
        # Initialize with default config if not already setup
        _logging_manager = LoggingManager()
    return _logging_manager.get_logger(name)


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging_config = {
        'level': 'DEBUG',
        'use_wandb': False,
        'log_file': 'test.log',
        'console_output': True
    }
    
    setup_logging(logging_config)
    logger = get_logger(__name__)
    
    # Test different log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test exception logging
    try:
        raise ValueError("This is a test exception")
    except Exception as e:
        logger.exception("Caught an exception")