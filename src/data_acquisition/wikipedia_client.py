"""
Wikipedia Client for PrismRAG

This module provides functionality to fetch real data from Wikipedia using
the Wikipedia API. It includes methods to search for pages, fetch content,
and process the data for use in training.
"""

import logging
import time
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

import wikipediaapi

# Initialize logger
logger = logging.getLogger(__name__)


class WikipediaCategory(Enum):
    """Categories for Wikipedia content filtering"""
    SCIENCE = "Science"
    HISTORY = "History"
    TECHNOLOGY = "Technology"
    CULTURE = "Culture"
    GEOGRAPHY = "Geography"
    BIOLOGY = "Biology"
    PHYSICS = "Physics"
    MATHEMATICS = "Mathematics"
    LITERATURE = "Literature"
    ART = "Art"


@dataclass
class WikipediaPage:
    """Data class for Wikipedia page content"""
    title: str
    content: str
    url: str
    categories: List[str]
    word_count: int
    page_id: int
    last_modified: str


class WikipediaClient:
    """
    Client for fetching and processing Wikipedia data.
    
    This client uses the wikipedia-api library to access real Wikipedia content
    with proper rate limiting and error handling.
    """
    
    def __init__(
        self,
        language: str = "en",
        user_agent: str = "PrismRAG/1.0 (https://github.com/xuanyue2019/prismrag; mkachuee@meta.com)",
        timeout: int = 30,
        rate_limit_delay: float = 1.0
    ):
        """
        Initialize the Wikipedia client.
        
        Args:
            language: Wikipedia language code (e.g., 'en', 'zh')
            user_agent: User agent string for API requests
            timeout: Request timeout in seconds
            rate_limit_delay: Delay between requests to respect rate limits
        """
        self.language = language
        self.user_agent = user_agent
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        
        # Initialize Wikipedia API client
        self.wiki = wikipediaapi.Wikipedia(
            language=language,
            user_agent=user_agent,
            timeout=timeout
        )
        
        logger.info(f"Wikipedia client initialized for {language} Wikipedia")
    
    def fetch_page(self, page_title: str) -> Optional[WikipediaPage]:
        """
        Fetch a single Wikipedia page by title.
        
        Args:
            page_title: Title of the Wikipedia page
            
        Returns:
            WikipediaPage object if successful, None otherwise
        """
        try:
            # Add delay to respect rate limits
            time.sleep(self.rate_limit_delay)
            
            page = self.wiki.page(page_title)
            
            if not page.exists():
                logger.warning(f"Page not found: {page_title}")
                return None
            
            # Extract categories
            categories = [cat.title for cat in page.categories.values()]
            
            wikipedia_page = WikipediaPage(
                title=page.title,
                content=page.text,
                url=page.fullurl,
                categories=categories,
                word_count=len(page.text.split()),
                page_id=page.pageid,
                last_modified=page.touched if hasattr(page, 'touched') else "unknown"
            )
            
            logger.debug(f"Fetched page: {page_title} ({wikipedia_page.word_count} words)")
            return wikipedia_page
            
        except Exception as e:
            logger.error(f"Error fetching page {page_title}: {e}")
            return None
    
    def search_pages(
        self,
        query: str,
        max_results: int = 10,
        min_word_count: int = 500,
        max_word_count: int = 7000
    ) -> List[WikipediaPage]:
        """
        Search for Wikipedia pages based on a query.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            min_word_count: Minimum word count for pages
            max_word_count: Maximum word count for pages
            
        Returns:
            List of WikipediaPage objects that match the criteria
        """
        try:
            # Add delay to respect rate limits
            time.sleep(self.rate_limit_delay)
            
            search_results = self.wiki.search(query, results=max_results)
            
            if not search_results:
                logger.warning(f"No search results for query: {query}")
                return []
            
            pages = []
            for result in search_results:
                page = self.fetch_page(result)
                if page and min_word_count <= page.word_count <= max_word_count:
                    pages.append(page)
                if len(pages) >= max_results:
                    break
            
            logger.info(f"Found {len(pages)} pages for query: {query}")
            return pages
            
        except Exception as e:
            logger.error(f"Error searching for '{query}': {e}")
            return []
    
    def get_pages_by_category(
        self,
        category: WikipediaCategory,
        max_pages: int = 50,
        min_word_count: int = 500,
        max_word_count: int = 7000
    ) -> List[WikipediaPage]:
        """
        Get pages from a specific Wikipedia category.
        
        Args:
            category: WikipediaCategory enum value
            max_pages: Maximum number of pages to return
            min_word_count: Minimum word count for pages
            max_word_count: Maximum word count for pages
            
        Returns:
            List of WikipediaPage objects in the category
        """
        try:
            category_page = self.wiki.page(f"Category:{category.value}")
            
            if not category_page.exists():
                logger.warning(f"Category not found: {category.value}")
                return []
            
            pages = []
            for page_title in category_page.categorymembers.keys():
                if len(pages) >= max_pages:
                    break
                
                page = self.fetch_page(page_title)
                if page and min_word_count <= page.word_count <= max_word_count:
                    pages.append(page)
            
            logger.info(f"Found {len(pages)} pages in category: {category.value}")
            return pages
            
        except Exception as e:
            logger.error(f"Error fetching category {category.value}: {e}")
            return []
    
    def get_random_pages(
        self,
        count: int = 10,
        min_word_count: int = 500,
        max_word_count: int = 7000
    ) -> List[WikipediaPage]:
        """
        Get random Wikipedia pages.
        
        Args:
            count: Number of random pages to fetch
            min_word_count: Minimum word count for pages
            max_word_count: Maximum word count for pages
            
        Returns:
            List of random WikipediaPage objects
        """
        # Note: wikipedia-api doesn't directly support random pages,
        # so we'll use the 'Special:Random' feature indirectly
        pages = []
        attempts = 0
        max_attempts = count * 3  # Allow some failures
        
        while len(pages) < count and attempts < max_attempts:
            attempts += 1
            try:
                # Fetch random page by accessing Special:Random
                random_page = self.wiki.page("Special:Random")
                
                if random_page.exists() and random_page.title not in [p.title for p in pages]:
                    page = self.fetch_page(random_page.title)
                    if page and min_word_count <= page.word_count <= max_word_count:
                        pages.append(page)
                        logger.debug(f"Added random page: {page.title}")
                
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.warning(f"Error fetching random page: {e}")
                continue
        
        logger.info(f"Fetched {len(pages)} random pages")
        return pages
    
    def batch_fetch_pages(
        self,
        page_titles: List[str],
        min_word_count: int = 500,
        max_word_count: int = 7000
    ) -> List[WikipediaPage]:
        """
        Fetch multiple pages in batch with error handling.
        
        Args:
            page_titles: List of page titles to fetch
            min_word_count: Minimum word count filter
            max_word_count: Maximum word count filter
            
        Returns:
            List of successfully fetched WikipediaPage objects
        """
        pages = []
        
        for title in page_titles:
            try:
                page = self.fetch_page(title)
                if page and min_word_count <= page.word_count <= max_word_count:
                    pages.append(page)
            except Exception as e:
                logger.warning(f"Failed to fetch page {title}: {e}")
                continue
        
        logger.info(f"Batch fetch completed: {len(pages)}/{len(page_titles)} pages successful")
        return pages


# Example usage and testing
if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize client
    client = WikipediaClient()
    
    # Test fetching a specific page
    page = client.fetch_page("Artificial intelligence")
    if page:
        print(f"Fetched page: {page.title}")
        print(f"Word count: {page.word_count}")
        print(f"Categories: {page.categories[:5]}")  # First 5 categories
    
    # Test searching
    search_results = client.search_pages("machine learning", max_results=3)
    for result in search_results:
        print(f"Search result: {result.title} ({result.word_count} words)")
    
    # Test category fetching
    science_pages = client.get_pages_by_category(WikipediaCategory.SCIENCE, max_pages=3)
    for page in science_pages:
        print(f"Science page: {page.title}")
    
    # Test random pages
    random_pages = client.get_random_pages(count=2)
    for page in random_pages:
        print(f"Random page: {page.title}")