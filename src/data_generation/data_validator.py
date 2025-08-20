"""
Data Validator for PrismRAG

Validates generated data samples against schema definitions, format requirements,
and quality standards to ensure consistency and reliability.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import jsonschema
from datetime import datetime

from src.utils import setup_logging


class ValidationLevel(Enum):
    """Validation levels for data samples"""
    CRITICAL = 3  # Must pass for data to be usable
    WARNING = 2   # Should pass but data may still be usable
    INFO = 1      # Optional validation for quality improvement


@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    validation_level: ValidationLevel
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]


class DataValidator:
    """
    Comprehensive data validator for PrismRAG generated data.
    
    Validates data samples against schema definitions, format requirements,
    and quality standards to ensure consistency and reliability.
    """
    
    def __init__(self):
        """Initialize the data validator"""
        self.logger = logging.getLogger(__name__)
        
        # Schema definitions
        self._initialize_schemas()
        
        # Validation rules
        self._initialize_validation_rules()
        
        self.logger.info("Data validator initialized")
    
    def _initialize_schemas(self) -> None:
        """Initialize JSON schemas for different data types"""
        
        # Schema for distractor samples
        self.distractor_schema = {
            "type": "object",
            "required": [
                "question", "answer", "golden_passage", "distractor_passage",
                "open_ended_question", "distractor_answer"
            ],
            "properties": {
                "question": {"type": "string", "minLength": 5},
                "answer": {"type": "string", "minLength": 1},
                "golden_passage": {"type": "string", "minLength": 50},
                "distractor_passage": {"type": "string", "minLength": 50},
                "open_ended_question": {"type": "string", "minLength": 5},
                "distractor_answer": {"type": "string", "minLength": 1},
                "user_time": {"type": ["string", "null"]},
                "location": {"type": ["string", "null"]},
                "quality_metrics": {"type": ["object", "null"]}
            }
        }
        
        # Schema for CoT samples
        self.cot_schema = {
            "type": "object",
            "required": [
                "question", "references", "strategy", "reasoning", "answer"
            ],
            "properties": {
                "question": {"type": "string", "minLength": 5},
                "references": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 50},
                    "minItems": 1
                },
                "strategy": {"type": "string", "minLength": 20},
                "reasoning": {"type": "string", "minLength": 50},
                "answer": {"type": "string", "minLength": 1},
                "user_context": {"type": ["string", "null"]},
                "quality_score": {"type": ["number", "null"]},
                "evaluation_feedback": {"type": ["string", "null"]}
            }
        }
        
        # Schema for seed QA samples
        self.seed_qa_schema = {
            "type": "object",
            "required": ["question", "answer", "passage"],
            "properties": {
                "question": {"type": "string", "minLength": 5},
                "answer": {"type": "string", "minLength": 1},
                "passage": {"type": "string", "minLength": 50},
                "source": {"type": ["string", "null"]},
                "metadata": {"type": ["object", "null"]}
            }
        }
    
    def _initialize_validation_rules(self) -> None:
        """Initialize validation rules and patterns"""
        
        # Common validation patterns
        self.validation_patterns = {
            'question_mark': r'[?？]$',  # Questions should end with question mark
            'reasonable_length': {
                'question': (5, 200),
                'answer': (1, 500),
                'passage': (50, 2000),
                'reasoning': (50, 1500)
            },
            'url_pattern': r'https?://\S+',
            'email_pattern': r'\S+@\S+\.\S+',
            'datetime_pattern': r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'
        }
        
        # Blacklisted patterns (common errors)
        self.blacklisted_patterns = [
            r'\[.*?\]',  # Placeholders like [something]
            r'\{.*?\}',  # Template variables
            r'\(.*?\)',  # Parenthetical notes
            r'[A-Z]{3,}',  # All caps (likely placeholder)
            r'xxx', r'xxx', r'###',  # Common placeholders
            r'fill in', r'enter here', r'your answer'  # Instruction text
        ]
    
    def validate_distractor_sample(self, sample: Dict) -> ValidationResult:
        """
        Validate a distractor sample against schema and quality rules.
        
        Args:
            sample: Distractor sample dictionary
            
        Returns:
            ValidationResult with validation details
        """
        errors = []
        warnings = []
        suggestions = []
        
        # Schema validation
        schema_result = self._validate_schema(sample, self.distractor_schema)
        errors.extend(schema_result.errors)
        warnings.extend(schema_result.warnings)
        
        # Content validation
        content_result = self._validate_distractor_content(sample)
        errors.extend(content_result.errors)
        warnings.extend(content_result.warnings)
        suggestions.extend(content_result.suggestions)
        
        # Quality metrics validation (if present)
        if 'quality_metrics' in sample and sample['quality_metrics']:
            quality_result = self._validate_quality_metrics(sample['quality_metrics'])
            warnings.extend(quality_result.warnings)
            suggestions.extend(quality_result.suggestions)
        
        is_valid = len([e for e in errors if self._is_critical_error(e)]) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            validation_level=ValidationLevel.CRITICAL if not is_valid else ValidationLevel.INFO,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def validate_cot_sample(self, sample: Dict) -> ValidationResult:
        """
        Validate a CoT sample against schema and quality rules.
        
        Args:
            sample: CoT sample dictionary
            
        Returns:
            ValidationResult with validation details
        """
        errors = []
        warnings = []
        suggestions = []
        
        # Schema validation
        schema_result = self._validate_schema(sample, self.cot_schema)
        errors.extend(schema_result.errors)
        warnings.extend(schema_result.warnings)
        
        # Content validation
        content_result = self._validate_cot_content(sample)
        errors.extend(content_result.errors)
        warnings.extend(content_result.warnings)
        suggestions.extend(content_result.suggestions)
        
        # Quality score validation (if present)
        if 'quality_score' in sample and sample['quality_score'] is not None:
            score_result = self._validate_quality_score(sample['quality_score'])
            warnings.extend(score_result.warnings)
            suggestions.extend(score_result.suggestions)
        
        is_valid = len([e for e in errors if self._is_critical_error(e)]) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            validation_level=ValidationLevel.CRITICAL if not is_valid else ValidationLevel.INFO,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def validate_seed_qa_sample(self, sample: Dict) -> ValidationResult:
        """
        Validate a seed QA sample against schema and quality rules.
        
        Args:
            sample: Seed QA sample dictionary
            
        Returns:
            ValidationResult with validation details
        """
        errors = []
        warnings = []
        suggestions = []
        
        # Schema validation
        schema_result = self._validate_schema(sample, self.seed_qa_schema)
        errors.extend(schema_result.errors)
        warnings.extend(schema_result.warnings)
        
        # Content validation
        content_result = self._validate_seed_qa_content(sample)
        errors.extend(content_result.errors)
        warnings.extend(content_result.warnings)
        suggestions.extend(content_result.suggestions)
        
        is_valid = len([e for e in errors if self._is_critical_error(e)]) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            validation_level=ValidationLevel.CRITICAL if not is_valid else ValidationLevel.INFO,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def _validate_schema(self, data: Dict, schema: Dict) -> ValidationResult:
        """Validate data against JSON schema"""
        errors = []
        warnings = []
        
        try:
            jsonschema.validate(instance=data, schema=schema)
        except jsonschema.ValidationError as e:
            errors.append(f"Schema validation failed: {e.message}")
        except Exception as e:
            errors.append(f"Schema validation error: {e}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            validation_level=ValidationLevel.CRITICAL,
            errors=errors,
            warnings=warnings,
            suggestions=[]
        )
    
    def _validate_distractor_content(self, sample: Dict) -> ValidationResult:
        """Validate distractor sample content"""
        errors = []
        warnings = []
        suggestions = []
        
        # Check for blacklisted patterns
        for field in ['question', 'answer', 'golden_passage', 'distractor_passage']:
            if field in sample:
                blacklist_result = self._check_blacklisted_patterns(sample[field], field)
                errors.extend(blacklist_result.errors)
                warnings.extend(blacklist_result.warnings)
        
        # Check question format
        if 'question' in sample and not re.search(self.validation_patterns['question_mark'], sample['question']):
            warnings.append("Question should end with a question mark")
            suggestions.append("Add a question mark at the end of the question")
        
        # Check answer consistency
        if 'answer' in sample and 'distractor_answer' in sample:
            if sample['answer'].lower() == sample['distractor_answer'].lower():
                errors.append("Distractor answer should be different from ground truth answer")
        
        # Check passage similarity (basic check)
        if 'golden_passage' in sample and 'distractor_passage' in sample:
            similarity = self._calculate_text_similarity(
                sample['golden_passage'], sample['distractor_passage']
            )
            if similarity > 0.8:
                warnings.append("Distractor passage is too similar to golden passage")
                suggestions.append("Increase the difference between passages")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            validation_level=ValidationLevel.WARNING,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def _validate_cot_content(self, sample: Dict) -> ValidationResult:
        """Validate CoT sample content"""
        errors = []
        warnings = []
        suggestions = []
        
        # Check for blacklisted patterns
        for field in ['question', 'strategy', 'reasoning', 'answer']:
            if field in sample:
                blacklist_result = self._check_blacklisted_patterns(sample[field], field)
                errors.extend(blacklist_result.errors)
                warnings.extend(blacklist_result.warnings)
        
        # Check reasoning structure
        if 'reasoning' in sample:
            reasoning = sample['reasoning']
            # Check if reasoning has step-by-step structure
            step_patterns = [r'步骤\d+', r'step \d+', r'-\s*步骤', r'-\s*step']
            has_steps = any(re.search(pattern, reasoning.lower()) for pattern in step_patterns)
            
            if not has_steps:
                warnings.append("Reasoning should have clear step-by-step structure")
                suggestions.append("Use numbered steps or bullet points in reasoning")
        
        # Check strategy-reasoning alignment
        if 'strategy' in sample and 'reasoning' in sample:
            strategy_keywords = set(re.findall(r'\b\w+\b', sample['strategy'].lower()))
            reasoning_keywords = set(re.findall(r'\b\w+\b', sample['reasoning'].lower()))
            
            overlap = strategy_keywords.intersection(reasoning_keywords)
            if len(overlap) < 3:  # At least 3 common keywords
                warnings.append("Strategy and reasoning may not be well aligned")
                suggestions.append("Ensure reasoning follows the outlined strategy")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            validation_level=ValidationLevel.WARNING,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def _validate_seed_qa_content(self, sample: Dict) -> ValidationResult:
        """Validate seed QA sample content"""
        errors = []
        warnings = []
        suggestions = []
        
        # Check for blacklisted patterns
        for field in ['question', 'answer', 'passage']:
            if field in sample:
                blacklist_result = self._check_blacklisted_patterns(sample[field], field)
                errors.extend(blacklist_result.errors)
                warnings.extend(blacklist_result.warnings)
        
        # Check answer in passage
        if 'answer' in sample and 'passage' in sample:
            answer = sample['answer'].lower()
            passage = sample['passage'].lower()
            
            if answer not in passage:
                warnings.append("Answer should be present in the passage")
                suggestions.append("Ensure the answer can be found in the passage text")
        
        # Check question-answer relevance
        if 'question' in sample and 'answer' in sample:
            question_words = set(re.findall(r'\b\w+\b', sample['question'].lower()))
            answer_words = set(re.findall(r'\b\w+\b', sample['answer'].lower()))
            
            overlap = question_words.intersection(answer_words)
            if len(overlap) < 2:  # At least 2 common words
                warnings.append("Question and answer may not be well related")
                suggestions.append("Ensure the answer directly addresses the question")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            validation_level=ValidationLevel.WARNING,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def _validate_quality_metrics(self, metrics: Dict) -> ValidationResult:
        """Validate quality metrics"""
        warnings = []
        suggestions = []
        
        expected_fields = ['relevance_score', 'distraction_score', 'format_score', 'overall_score']
        for field in expected_fields:
            if field not in metrics:
                warnings.append(f"Missing quality metric: {field}")
                continue
            
            score = metrics[field]
            if not isinstance(score, (int, float)) or not (1 <= score <= 5):
                warnings.append(f"Invalid {field}: {score} (should be 1-5)")
        
        return ValidationResult(
            is_valid=True,
            validation_level=ValidationLevel.INFO,
            errors=[],
            warnings=warnings,
            suggestions=suggestions
        )
    
    def _validate_quality_score(self, score: float) -> ValidationResult:
        """Validate quality score"""
        warnings = []
        suggestions = []
        
        if not isinstance(score, (int, float)) or not (1 <= score <= 4):
            warnings.append(f"Invalid quality score: {score} (should be 1-4)")
        elif score < 3:
            suggestions.append("Consider improving the sample quality")
        
        return ValidationResult(
            is_valid=True,
            validation_level=ValidationLevel.INFO,
            errors=[],
            warnings=warnings,
            suggestions=suggestions
        )
    
    def _check_blacklisted_patterns(self, text: str, field_name: str) -> ValidationResult:
        """Check for blacklisted patterns in text"""
        errors = []
        warnings = []
        
        for pattern in self.blacklisted_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                errors.append(f"Blacklisted pattern found in {field_name}: {pattern}")
        
        # Check for URLs and emails (usually not desired in generated data)
        if re.search(self.validation_patterns['url_pattern'], text):
            warnings.append(f"URL found in {field_name}")
        
        if re.search(self.validation_patterns['email_pattern'], text):
            warnings.append(f"Email found in {field_name}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            validation_level=ValidationLevel.CRITICAL if errors else ValidationLevel.WARNING,
            errors=errors,
            warnings=warnings,
            suggestions=[]
        )
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate basic text similarity (0-1)"""
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _is_critical_error(self, error: str) -> bool:
        """Check if an error is critical"""
        critical_keywords = [
            'required', 'missing', 'invalid', 'blacklisted', 'schema'
        ]
        return any(keyword in error.lower() for keyword in critical_keywords)
    
    def validate_batch(
        self,
        samples: List[Dict],
        sample_type: str = "distractor"
    ) -> List[Tuple[Dict, ValidationResult]]:
        """
        Validate a batch of samples.
        
        Args:
            samples: List of sample dictionaries
            sample_type: Type of samples ("distractor", "cot", or "seed_qa")
            
        Returns:
            List of (sample, validation_result) tuples
        """
        results = []
        
        for i, sample in enumerate(samples):
            self.logger.info(f"Validating sample {i + 1}/{len(samples)}")
            
            try:
                if sample_type == "distractor":
                    result = self.validate_distractor_sample(sample)
                elif sample_type == "cot":
                    result = self.validate_cot_sample(sample)
                elif sample_type == "seed_qa":
                    result = self.validate_seed_qa_sample(sample)
                else:
                    result = ValidationResult(
                        is_valid=False,
                        validation_level=ValidationLevel.CRITICAL,
                        errors=[f"Unknown sample type: {sample_type}"],
                        warnings=[],
                        suggestions=[]
                    )
                
                results.append((sample, result))
                
            except Exception as e:
                self.logger.error(f"Error validating sample {i + 1}: {e}")
                error_result = ValidationResult(
                    is_valid=False,
                    validation_level=ValidationLevel.CRITICAL,
                    errors=[f"Validation error: {e}"],
                    warnings=[],
                    suggestions=[]
                )
                results.append((sample, error_result))
        
        return results
    
    def generate_validation_report(
        self,
        validation_results: List[Tuple[Dict, ValidationResult]]
    ) -> Dict:
        """
        Generate comprehensive validation report.
        
        Args:
            validation_results: List of (sample, validation_result) tuples
            
        Returns:
            Validation report dictionary
        """
        total_samples = len(validation_results)
        valid_samples = sum(1 for _, result in validation_results if result.is_valid)
        invalid_samples = total_samples - valid_samples
        
        error_counts = {}
        warning_counts = {}
        
        for _, result in validation_results:
            for error in result.errors:
                error_type = error.split(':')[0] if ':' in error else 'Other'
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            for warning in result.warnings:
                warning_type = warning.split(':')[0] if ':' in warning else 'Other'
                warning_counts[warning_type] = warning_counts.get(warning_type, 0) + 1
        
        return {
            "total_samples": total_samples,
            "valid_samples": valid_samples,
            "invalid_samples": invalid_samples,
            "validation_rate": f"{(valid_samples / total_samples * 100):.1f}%" if total_samples > 0 else "0%",
            "error_types": error_counts,
            "warning_types": warning_counts,
            "sample_breakdown": [
                {
                    "sample_index": i,
                    "is_valid": result.is_valid,
                    "error_count": len(result.errors),
                    "warning_count": len(result.warnings),
                    "suggestion_count": len(result.suggestions)
                }
                for i, (_, result) in enumerate(validation_results)
            ]
        }
    
    def save_validation_report(
        self,
        validation_results: List[Tuple[Dict, ValidationResult]],
        filepath: str
    ) -> None:
        """
        Save validation report to JSON file.
        
        Args:
            validation_results: List of (sample, validation_result) tuples
            filepath: Path to save the report
        """
        report = self.generate_validation_report(validation_results)
        
        from pathlib import Path
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Validation report saved to {filepath}")


# Example usage
if __name__ == "__main__":
    # Setup logging
    setup_logging()
    
    # Initialize validator
    validator = DataValidator()
    
    # Example distractor sample
    distractor_sample = {
        "question": "What is the capital of France?",
        "answer": "Paris",
        "golden_passage": "France is a country in Western Europe. Its capital is Paris, known for the Eiffel Tower and Louvre Museum.",
        "distractor_passage": "France is a nation in Western Europe. Its main city is Lyon, famous for its culinary traditions and historical architecture.",
        "open_ended_question": "Which city serves as the capital of France?",
        "distractor_answer": "Lyon",
        "user_time": "2024-01-15 14:30:00",
        "location": "Europe"
    }
    
    # Validate the sample
    result = validator.validate_distractor_sample(distractor_sample)
    
    print(f"Validation result: {'VALID' if result.is_valid else 'INVALID'}")
    print(f"Errors: {result.errors}")
    print(f"Warnings: {result.warnings}")
    print(f"Suggestions: {result.suggestions}")