"""
Seed Data Generator for PrismRAG

This module implements the seed QA generation mechanism that creates
question-answer-passage triplets from real data sources following the
inverse problem approach described in the PrismRAG paper.
"""

import logging
import json
import random
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.data_acquisition import WikipediaClient, WebSearchClient
from src.utils import ConfigManager, setup_logging

# Initialize logger
logger = logging.getLogger(__name__)


@dataclass
class SeedSample:
    """Data class for seed samples containing QA pairs and source passages"""
    question: str
    answer: str
    passage: str
    source: str  # "wikipedia" or "web_search"
    metadata: Optional[Dict] = None


class SeedDataGenerator:
    """
    Generates seed QA data from real data sources.
    
    This class implements the inverse problem approach: given document chunks,
    generate question-answer pairs based on their content using LLMs.
    """
    
    def __init__(
        self,
        config: Optional[ConfigManager] = None,
        model_name: Optional[str] = None,
        device: str = "auto"
    ):
        """
        Initialize the seed data generator.
        
        Args:
            config: Configuration manager instance
            model_name: Name of the LLM model to use for QA generation
            device: Device to run the model on ("auto", "cuda", "cpu")
        """
        self.config = config or ConfigManager()
        self.model_name = model_name or self.config.get('model.base_model')
        self.device = device
        
        # Initialize data acquisition clients
        self.wikipedia_client = WikipediaClient()
        self.web_search_client = WebSearchClient()
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Load model and tokenizer
        self._load_model()
        
        self.logger.info(f"Seed data generator initialized with model: {self.model_name}")
    
    def _load_model(self) -> None:
        """Load the LLM model and tokenizer for QA generation"""
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
    
    def generate_from_wikipedia(
        self,
        max_pages: int = 100,
        min_words: int = 500,
        max_words: int = 7000,
        min_chunk_words: int = 250,
        max_chunk_words: int = 1000,
        num_references_min: int = 2,
        num_references_max: int = 15
    ) -> List[SeedSample]:
        """
        Generate seed data from Wikipedia pages.
        
        Args:
            max_pages: Maximum number of Wikipedia pages to process
            min_words: Minimum words per page
            max_words: Maximum words per page
            min_chunk_words: Minimum words per chunk
            max_chunk_words: Maximum words per chunk
            num_references_min: Minimum number of reference chunks
            num_references_max: Maximum number of reference chunks
            
        Returns:
            List of SeedSample objects
        """
        self.logger.info("Generating seed data from Wikipedia...")
        
        seed_samples = []
        
        # Get random Wikipedia pages
        wikipedia_pages = self.wikipedia_client.get_random_pages(
            count=max_pages,
            min_word_count=min_words,
            max_word_count=max_words
        )
        
        for page in wikipedia_pages:
            try:
                # Split page into chunks
                chunks = self._split_into_chunks(
                    page.content,
                    min_size=min_chunk_words,
                    max_size=max_chunk_words
                )
                
                if len(chunks) < num_references_min:
                    continue
                
                # Randomly select chunks as references
                num_refs = random.randint(
                    num_references_min, 
                    min(num_references_max, len(chunks))
                )
                selected_chunks = random.sample(chunks, num_refs)
                
                # Select one chunk as the golden reference for QA generation
                golden_chunk = random.choice(selected_chunks)
                
                # Generate QA pair from golden chunk
                qa_pair = self._generate_qa_from_chunk(golden_chunk)
                
                if qa_pair and qa_pair["question"] != "N/A" and qa_pair["answer"] != "N/A":
                    seed_samples.append(SeedSample(
                        question=qa_pair["question"],
                        answer=qa_pair["answer"],
                        passage=golden_chunk,
                        source="wikipedia",
                        metadata={
                            "num_references": len(selected_chunks),
                            "all_references": selected_chunks,
                            "word_count": page.word_count,
                            "page_title": page.title,
                            "page_url": page.url,
                            "categories": page.categories
                        }
                    ))
                    
                    self.logger.debug(f"Generated QA from Wikipedia: {qa_pair['question'][:50]}...")
                
            except Exception as e:
                self.logger.warning(f"Error processing Wikipedia page {page.title}: {e}")
                continue
        
        self.logger.info(f"Generated {len(seed_samples)} seed samples from Wikipedia")
        return seed_samples
    
    def generate_from_web_search(
        self,
        queries: List[str],
        max_results_per_query: int = 10,
        max_words_per_page: int = 3000
    ) -> List[SeedSample]:
        """
        Generate seed data from web search results.
        
        Args:
            queries: List of search queries to use
            max_results_per_query: Maximum results per query
            max_words_per_page: Maximum words per page
            
        Returns:
            List of SeedSample objects
        """
        self.logger.info("Generating seed data from web search...")
        
        seed_samples = []
        
        # Perform batch search
        search_results = self.web_search_client.batch_search(
            queries=queries,
            max_results_per_query=max_results_per_query
        )
        
        for query, results in search_results.items():
            for result in results:
                try:
                    # Truncate content if too long
                    words = result.content.split()
                    if len(words) > max_words_per_page:
                        content = " ".join(words[:max_words_per_page])
                    else:
                        content = result.content
                    
                    # Generate QA pair from content
                    qa_pair = self._generate_qa_from_chunk(
                        content,
                        context={
                            "query": query,
                            "time": result.timestamp,
                            "location": "unknown"  # Could be enhanced with geoIP
                        }
                    )
                    
                    if qa_pair and qa_pair["question"] != "N/A" and qa_pair["answer"] != "N/A":
                        seed_samples.append(SeedSample(
                            question=qa_pair["question"],
                            answer=qa_pair["answer"],
                            passage=content,
                            source="web_search",
                            metadata={
                                "original_query": query,
                                "url": result.url,
                                "source": result.source,
                                "timestamp": result.timestamp,
                                "word_count": len(content.split())
                            }
                        ))
                        
                        self.logger.debug(f"Generated QA from web search: {qa_pair['question'][:50]}...")
                    
                except Exception as e:
                    self.logger.warning(f"Error processing web result {result.url}: {e}")
                    continue
        
        self.logger.info(f"Generated {len(seed_samples)} seed samples from web search")
        return seed_samples
    
    def _split_into_chunks(
        self,
        text: str,
        min_size: int,
        max_size: int
    ) -> List[str]:
        """Split text into non-overlapping chunks of appropriate size"""
        words = text.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            
            if len(current_chunk) >= max_size:
                if len(current_chunk) >= min_size:
                    chunks.append(" ".join(current_chunk))
                current_chunk = []
        
        # Add remaining chunk if it meets minimum size
        if len(current_chunk) >= min_size:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _generate_qa_from_chunk(
        self,
        chunk: str,
        context: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Generate a question-answer pair from a text chunk using LLM"""
        prompt = self._build_qa_generation_prompt(chunk, context)
        
        try:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=3000
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    temperature=0.8,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )
            
            return self._parse_qa_response(response)
            
        except Exception as e:
            self.logger.error(f"Error generating QA from chunk: {e}")
            return None
    
    def _build_qa_generation_prompt(
        self,
        content: str,
        context: Optional[Dict] = None
    ) -> str:
        """Build prompt for QA generation based on paper's appendix"""
        prompt = """You are a helpful assistant. Always follow the provided instructions and generate outputs in valid json format without any extra information. Generate a question and answer pair based on the Provided Content below.

## Requirements:
- You must ground your question and answer to the Provided Content
- The question should be selected to resemble what a curious college graduate would ask an intelligent conversational system. From the difficulty level of 1 to 10, aim for an 8.
- The answer should be fully and directly grounded on the Provided Content. Never use any information other than what is available in the Provided Content to generate the question and answer.
- Never generate a question that is asking for the current time, date, or location.
- The question should not be too general or vague. When applicable, include specific entities, names, times, locations, events, and keywords in the question.
- The question and answer must be grammatically correct and be conversationally natural.
- A good question should be meaningful and provides enough context.
- Always return in json format with two keys: "question" and "answer". If the Provided Content is not readable, you may set the value corresponding to the question and answer keys to "N/A".

## Examples:
Here are some examples of questions types to consider:
1. How old is Obama?
2. What was the name of the first president of the United States?
3. How is the weather in Seattle this weekend?
4. What is the population of China?
5. Is there a movie theater nearby?
6. What time is high tide tonight in Santa Cruz, CA?
7. Who is leading in the election between Trump and Kamala Harris?
8. What is the dodgers current score?
9. When does daylight saving time start?
10. Any updates on Morgan Freeman's health?
11. What are the main ingredients in a margarita?
12. Why is chocolate bad for dogs?

## Provided Content:
{content}"""

        if context:
            prompt += f"\n\n## Context Information:\n"
            for key, value in context.items():
                if value:
                    prompt += f"- {key}: {value}\n"
        
        return prompt.format(content=content[:2000])  # Limit content length
    
    def _parse_qa_response(self, response: str) -> Optional[Dict]:
        """Parse QA response from LLM output"""
        try:
            # Find JSON content
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                return None
            
            json_str = response[start_idx:end_idx]
            qa_data = json.loads(json_str)
            
            # Validate required fields
            if "question" not in qa_data or "answer" not in qa_data:
                return None
            
            # Check for N/A responses
            if qa_data["question"] == "N/A" or qa_data["answer"] == "N/A":
                return None
            
            return qa_data
            
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"Failed to parse QA response: {e}")
            return None
    
    def generate_batch(
        self,
        wikipedia_config: Optional[Dict] = None,
        web_search_config: Optional[Dict] = None
    ) -> Dict[str, List[SeedSample]]:
        """
        Generate seed data from both Wikipedia and web search sources.
        
        Args:
            wikipedia_config: Configuration for Wikipedia data generation
            web_search_config: Configuration for web search data generation
            
        Returns:
            Dictionary with Wikipedia and web search samples
        """
        wikipedia_config = wikipedia_config or {}
        web_search_config = web_search_config or {}
        
        # Default web search queries if not provided
        default_queries = [
            "latest technology news",
            "scientific discoveries 2024",
            "historical events this week",
            "current sports results",
            "recent medical breakthroughs"
        ]
        
        # Generate from Wikipedia
        wiki_samples = self.generate_from_wikipedia(**wikipedia_config)
        
        # Generate from web search
        web_queries = web_search_config.get('queries', default_queries)
        web_samples = self.generate_from_web_search(
            queries=web_queries,
            max_results_per_query=web_search_config.get('max_results_per_query', 10),
            max_words_per_page=web_search_config.get('max_words_per_page', 3000)
        )
        
        return {
            "wikipedia": wiki_samples,
            "web_search": web_samples
        }
    
    def save_samples(self, samples: List[SeedSample], filepath: str) -> None:
        """Save seed samples to JSON file"""
        data = []
        for sample in samples:
            data.append({
                "question": sample.question,
                "answer": sample.answer,
                "passage": sample.passage,
                "source": sample.source,
                "metadata": sample.metadata
            })
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Saved {len(samples)} seed samples to {filepath}")
    
    def load_samples(self, filepath: str) -> List[SeedSample]:
        """Load seed samples from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        for item in data:
            samples.append(SeedSample(
                question=item["question"],
                answer=item["answer"],
                passage=item["passage"],
                source=item["source"],
                metadata=item.get("metadata")
            ))
        
        self.logger.info(f"Loaded {len(samples)} seed samples from {filepath}")
        return samples


# Example usage
if __name__ == "__main__":
    # Setup logging
    setup_logging()
    
    # Initialize generator
    generator = SeedDataGenerator()
    
    # Generate samples from Wikipedia
    wiki_samples = generator.generate_from_wikipedia(max_pages=5)
    
    # Generate samples from web search
    web_samples = generator.generate_from_web_search(
        queries=["artificial intelligence advancements", "quantum computing progress"],
        max_results_per_query=3
    )
    
    # Save samples
    if wiki_samples:
        generator.save_samples(wiki_samples, "data/seed_wikipedia.json")
    
    if web_samples:
        generator.save_samples(web_samples, "data/seed_web_search.json")
    
    print(f"Generated {len(wiki_samples)} Wikipedia samples and {len(web_samples)} web search samples")