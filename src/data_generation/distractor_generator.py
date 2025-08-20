"""
Distractor Generator for PrismRAG

This module implements the distractor generation mechanism that creates
synthetic distractors by modifying named entities, locations, and temporal
information in golden passages, following the methodology described in the
PrismRAG paper.
"""

import logging
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.utils import ConfigManager, setup_logging

# Initialize logger
logger = logging.getLogger(__name__)


class DistractorType(Enum):
    """Types of distractors based on modification strategy"""
    NAMED_ENTITY = "named_entity"
    TEMPORAL = "temporal"
    LOCATION = "location"
    NUMERICAL = "numerical"
    HYBRID = "hybrid"


@dataclass
class DistractorSample:
    """Data class for distractor samples"""
    question: str
    answer: str
    golden_passage: str
    open_ended_question: str
    distractor_passage: str
    distractor_answer: str
    distractor_type: DistractorType
    user_time: Optional[str] = None
    location: Optional[str] = None
    quality_scores: Optional[Dict[str, float]] = None


class DistractorGenerator:
    """
    Generates synthetic distractors for training data.
    
    Based on the PrismRAG paper's approach of systematically modifying
    named entities, locations, and temporal information to create
    confusing but grammatically coherent distractor passages.
    """
    
    def __init__(
        self,
        config: Optional[ConfigManager] = None,
        model_name: Optional[str] = None,
        device: str = "auto",
        max_iterations: int = 5,
        quality_threshold: float = 4.0
    ):
        """
        Initialize the distractor generator.
        
        Args:
            config: Configuration manager instance
            model_name: Name of the LLM model to use for distractor generation
            device: Device to run the model on ("auto", "cuda", "cpu")
            max_iterations: Maximum number of generation iterations
            quality_threshold: Minimum quality score threshold for acceptance
        """
        self.config = config or ConfigManager()
        self.model_name = model_name or self.config.get('model.base_model')
        self.device = device
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Load model and tokenizer
        self._load_model()
        
        self.logger.info(f"Distractor generator initialized with model: {self.model_name}")
    
    def _load_model(self) -> None:
        """Load the LLM model and tokenizer for distractor generation"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.logger.info(f"Model {self.model_name} loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def generate_distractor(
        self,
        question: str,
        answer: str,
        passage: str,
        user_time: Optional[str] = None,
        location: Optional[str] = None,
        distractor_type: Optional[DistractorType] = None
    ) -> Optional[DistractorSample]:
        """
        Generate a distractor sample for the given question-answer-passage triplet.
        
        Args:
            question: Original question
            answer: Ground truth answer
            passage: Golden passage containing the answer
            user_time: User's current time context
            location: User's location context
            distractor_type: Type of distractor to generate
            
        Returns:
            DistractorSample if successful, None otherwise
        """
        self.logger.info(f"Generating distractor for question: {question[:50]}...")
        
        # Determine distractor type if not specified
        if distractor_type is None:
            distractor_type = self._determine_distractor_type(question, answer, passage)
        
        prior_distractor = None
        prior_rejection_reason = None
        
        for iteration in range(self.max_iterations):
            try:
                # Generate distractor using LLM
                distractor_data = self._generate_distractor_with_llm(
                    question=question,
                    answer=answer,
                    passage=passage,
                    user_time=user_time,
                    location=location,
                    distractor_type=distractor_type,
                    prior_distractor=prior_distractor,
                    prior_rejection_reason=prior_rejection_reason
                )
                
                if not distractor_data:
                    continue
                
                # Evaluate the generated distractor
                evaluation = self._evaluate_distractor(
                    question=question,
                    answer=answer,
                    passage=passage,
                    user_time=user_time,
                    location=location,
                    open_ended_question=distractor_data["open_ended_question"],
                    distractor_passage=distractor_data["distractor_passage"],
                    distractor_answer=distractor_data.get("distractor_answer", "")
                )
                
                # Check if quality meets threshold
                if evaluation["overall_score"] >= self.quality_threshold:
                    return DistractorSample(
                        question=question,
                        answer=answer,
                        golden_passage=passage,
                        open_ended_question=distractor_data["open_ended_question"],
                        distractor_passage=distractor_data["distractor_passage"],
                        distractor_answer=distractor_data.get("distractor_answer", ""),
                        distractor_type=distractor_type,
                        user_time=user_time,
                        location=location,
                        quality_scores={
                            "relevance_score": evaluation.get("relevance_score", 0),
                            "distraction_score": evaluation.get("distraction_score", 0),
                            "format_score": evaluation.get("format_score", 0),
                            "overall_score": evaluation.get("overall_score", 0)
                        }
                    )
                
                # Use evaluation feedback for next iteration
                prior_distractor = distractor_data["distractor_passage"]
                prior_rejection_reason = evaluation.get("feedback", "Quality below threshold")
                
                self.logger.debug(f"Iteration {iteration + 1}: Score {evaluation['overall_score']}, Feedback: {prior_rejection_reason}")
                
            except Exception as e:
                self.logger.warning(f"Error in iteration {iteration + 1}: {e}")
                continue
        
        self.logger.warning(f"Failed to generate quality distractor after {self.max_iterations} iterations")
        return None
    
    def _determine_distractor_type(
        self,
        question: str,
        answer: str,
        passage: str
    ) -> DistractorType:
        """Determine the most appropriate distractor type based on content analysis using LLM"""
        # Use LLM to analyze content and identify key elements
        content_analysis = self._analyze_content_with_llm(question, answer, passage)
        
        # Prioritize distractor types based on content analysis
        if content_analysis.get('has_temporal', False):
            return DistractorType.TEMPORAL
        elif content_analysis.get('has_locations', False):
            return DistractorType.LOCATION
        elif content_analysis.get('has_named_entities', False):
            return DistractorType.NAMED_ENTITY
        elif content_analysis.get('has_numerical', False):
            return DistractorType.NUMERICAL
        else:
            return DistractorType.HYBRID
    
    def _analyze_content_with_llm(
        self,
        question: str,
        answer: str,
        passage: str
    ) -> Dict[str, bool]:
        """Analyze content using LLM to identify key elements for distractor generation"""
        full_text = f"Question: {question}\nAnswer: {answer}\nPassage: {passage}"
        
        prompt = f"""Analyze the following text and identify if it contains temporal information, locations, named entities, and numerical values. 
Return your analysis in JSON format with boolean values for keys: "has_temporal", "has_locations", "has_named_entities", "has_numerical".

Text:
{full_text}

JSON:"""
        
        try:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2000
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )
            
            # Parse JSON response
            analysis = self._parse_json_response(response)
            if analysis:
                return {
                    'has_temporal': analysis.get('has_temporal', False),
                    'has_locations': analysis.get('has_locations', False),
                    'has_named_entities': analysis.get('has_named_entities', False),
                    'has_numerical': analysis.get('has_numerical', False)
                }
            
        except Exception as e:
            self.logger.error(f"Error analyzing content with LLM: {e}")
        
        # Fallback to default analysis if LLM fails
        return {
            'has_temporal': False,
            'has_locations': False,
            'has_named_entities': False,
            'has_numerical': False
        }
    
    def _generate_distractor_with_llm(
        self,
        question: str,
        answer: str,
        passage: str,
        user_time: Optional[str],
        location: Optional[str],
        distractor_type: DistractorType,
        prior_distractor: Optional[str] = None,
        prior_rejection_reason: Optional[str] = None
    ) -> Optional[Dict]:
        """Generate distractor using LLM with structured prompt"""
        prompt = self._build_distractor_generation_prompt(
            question=question,
            answer=answer,
            passage=passage,
            user_time=user_time,
            location=location,
            distractor_type=distractor_type,
            prior_distractor=prior_distractor,
            prior_rejection_reason=prior_rejection_reason
        )
        
        try:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=4000
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1000,
                    temperature=1.0,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )
            
            # Parse JSON response
            return self._parse_json_response(response)
            
        except Exception as e:
            self.logger.error(f"Error generating distractor: {e}")
            return None
    
    def _build_distractor_generation_prompt(
        self,
        question: str,
        answer: str,
        passage: str,
        user_time: Optional[str],
        location: Optional[str],
        distractor_type: DistractorType,
        prior_distractor: Optional[str] = None,
        prior_rejection_reason: Optional[str] = None
    ) -> str:
        """Build the prompt for distractor generation based on paper's appendix"""
        prompt = f"""You are an intelligent assistant with expertise in linguistics. Always follow the provided instructions and generate outputs in valid json format without any extra information.

### User:
Given a question, answer, passage, location and user-time, identify relevant named-entities, date times, locations in the passage based on the question and answer, such that modifying the relevant named entities will result in a new passage that can cause confusion to the user if they didn't have enough context.

Generate an open ended question "open-ended-question" by modifying the provided question such that the answer provided answers the new open ended question.

Now using question, answer, user time, location, modify the passage and generate a new passage by modifying the named entities such that:
a. The new passage is relevant to the existing passage.
b. The new passage is grammatically coherent.
c. For the "open-ended-question", both provided and your generated passage are relevant.
d. The distraction-passage should have a similar number of characters as the original passage and similar format.

## Requirements:
- You must generate the passage based on the user question, location, answer, user-time.
- The generated distraction should be of similar length to the original passage and with similar special characters such as \\n, \\t. Do not reduce the total number of words.
- Think it through step by step and provide a detailed explanation in the "thought-steps" field.
- Output in Json with following fields: 'open-ended-question', 'thought-steps', 'distracting-named-entities', 'distractor-passage', 'score', 'reason'

## Distractor Type: {distractor_type.value}
## Question: {question}
## Answer: {answer}
## User Time: {user_time or 'Not provided'}
## Location: {location or 'Not provided'}
## Passage: {passage}"""

        if prior_distractor and prior_rejection_reason:
            prompt += f"""
## Prior Distraction Passage: {prior_distractor}
## Prior Distractor Rejecting Reason: {prior_rejection_reason}"""

        prompt += "\n\n### Assistant:"
        
        return prompt
    
    def _evaluate_distractor(
        self,
        question: str,
        answer: str,
        passage: str,
        user_time: Optional[str],
        location: Optional[str],
        open_ended_question: str,
        distractor_passage: str,
        distractor_answer: str
    ) -> Dict:
        """Evaluate the quality of generated distractor"""
        prompt = self._build_distractor_evaluation_prompt(
            question=question,
            answer=answer,
            passage=passage,
            user_time=user_time,
            location=location,
            open_ended_question=open_ended_question,
            distractor_passage=distractor_passage,
            distractor_answer=distractor_answer
        )
        
        try:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=4000
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )
            
            evaluation = self._parse_json_response(response)
            
            if evaluation:
                # Calculate overall score as weighted average
                scores = {
                    "relevance_score": evaluation.get("relevance-score", 1),
                    "distraction_score": evaluation.get("distraction-score", 1),
                    "format_score": evaluation.get("format-score", 1)
                }
                
                # Weighted average using configurable weights
                weights = {
                    "relevance_score": self.config.get('data_generation.distractor.relevance_weight', 0.4),
                    "distraction_score": self.config.get('data_generation.distractor.distraction_weight', 0.4),
                    "format_score": self.config.get('data_generation.distractor.format_weight', 0.2)
                }
                
                overall_score = sum(scores[key] * weights[key] for key in scores.keys())
                evaluation["overall_score"] = overall_score
                evaluation["feedback"] = evaluation.get("thought-process", "")
            
            return evaluation or {
                "relevance_score": 1,
                "distraction_score": 1,
                "format_score": 1,
                "overall_score": 1,
                "feedback": "Failed to parse evaluation"
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating distractor: {e}")
            return {
                "relevance_score": 1,
                "distraction_score": 1,
                "format_score": 1,
                "overall_score": 1,
                "feedback": f"Evaluation error: {e}"
            }
    
    def _build_distractor_evaluation_prompt(
        self,
        question: str,
        answer: str,
        passage: str,
        user_time: Optional[str],
        location: Optional[str],
        open_ended_question: str,
        distractor_passage: str,
        distractor_answer: str
    ) -> str:
        """Build prompt for distractor evaluation based on paper's appendix"""
        return f"""You are an intelligent assistant with expertise in linguistics. Always follow the provided instructions and generate outputs in valid json format without any extra information.

### User:
Given a question, answer, passage, location, user-time, distraction-passage and distraction passage's answer:

Score on the scale of 1 to 5 on Scores:

1. relevance-score:
Measures how relevant the distraction-passage is to the given question, answer, passage, location, user-time.

2. distraction-score:
Measures how much of a distraction the passage is to a user who asked the question when provided with both passage and distraction-passage.

3. format-score:
Measures how similar in text length and format, the distraction passage is, with respect to the original passage.

Output in Json with following fields: 'relevance-score', 'distraction-score', 'format-score', 'thought-process'

## Question: {question}
## Answer: {answer}
## User Time: {user_time or 'Not provided'}
## Location: {location or 'Not provided'}
## Passage: {passage}
## Open Ended Question: {open_ended_question}
## Distraction Passage: {distractor_passage}
## Distraction Passage's Answer: {distractor_answer}

### Assistant:"""
    
    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """Parse JSON response from LLM"""
        try:
            # Find JSON content between braces
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                return None
            
            json_str = response[start_idx:end_idx]
            return json.loads(json_str)
            
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"Failed to parse JSON response: {e}")
            return None
    
    def generate_batch(
        self,
        samples: List[Tuple[str, str, str]],
        user_time: Optional[str] = None,
        location: Optional[str] = None
    ) -> List[DistractorSample]:
        """
        Generate distractors for a batch of samples.
        
        Args:
            samples: List of (question, answer, passage) tuples
            user_time: User's current time context
            location: User's location context
            
        Returns:
            List of DistractorSample objects
        """
        results = []
        
        for i, (question, answer, passage) in enumerate(samples):
            self.logger.info(f"Processing sample {i + 1}/{len(samples)}")
            
            distractor = self.generate_distractor(
                question=question,
                answer=answer,
                passage=passage,
                user_time=user_time,
                location=location
            )
            
            if distractor:
                results.append(distractor)
                self.logger.debug(f"Successfully generated distractor for sample {i + 1}")
            else:
                self.logger.warning(f"Failed to generate distractor for sample {i + 1}")
        
        self.logger.info(f"Generated {len(results)} distractor samples from {len(samples)} input samples")
        return results
    
    def save_samples(self, samples: List[DistractorSample], filepath: str) -> None:
        """Save distractor samples to JSON file"""
        data = []
        for sample in samples:
            data.append({
                "question": sample.question,
                "answer": sample.answer,
                "golden_passage": sample.golden_passage,
                "open_ended_question": sample.open_ended_question,
                "distractor_passage": sample.distractor_passage,
                "distractor_answer": sample.distractor_answer,
                "distractor_type": sample.distractor_type.value,
                "user_time": sample.user_time,
                "location": sample.location,
                "quality_scores": sample.quality_scores
            })
        
        # Ensure directory exists
        from pathlib import Path
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Saved {len(samples)} distractor samples to {filepath}")
    
    def load_samples(self, filepath: str) -> List[DistractorSample]:
        """Load distractor samples from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        for item in data:
            samples.append(DistractorSample(
                question=item["question"],
                answer=item["answer"],
                golden_passage=item["golden_passage"],
                open_ended_question=item["open_ended_question"],
                distractor_passage=item["distractor_passage"],
                distractor_answer=item["distractor_answer"],
                distractor_type=DistractorType(item["distractor_type"]),
                user_time=item.get("user_time"),
                location=item.get("location"),
                quality_scores=item.get("quality_scores")
            ))
        
        self.logger.info(f"Loaded {len(samples)} distractor samples from {filepath}")
        return samples


# Example usage
if __name__ == "__main__":
    # Setup logging
    setup_logging()
    
    # Initialize generator
    generator = DistractorGenerator()
    
    # Example QA passage triplet
    sample_question = "What is the capital of France?"
    sample_answer = "Paris"
    sample_passage = "France is a country located in Western Europe. Its capital city is Paris, which is known for the Eiffel Tower and the Louvre Museum. Paris has a population of about 2.2 million people."
    
    # Generate distractor
    distractor = generator.generate_distractor(
        question=sample_question,
        answer=sample_answer,
        passage=sample_passage,
        user_time="2024-01-19 14:30:00",
        location="New York"
    )
    
    if distractor:
        print(f"Generated distractor:")
        print(f"Open-ended question: {distractor.open_ended_question}")
        print(f"Distractor passage: {distractor.distractor_passage[:100]}...")
        print(f"Quality scores: {distractor.quality_scores}")
    else:
        print("Failed to generate distractor")