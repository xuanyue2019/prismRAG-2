"""
Web Search Client for PrismRAG

This module provides functionality to fetch real data from web search engines
using APIs like SerpAPI or Google Custom Search API. It includes methods to
perform searches, extract relevant content, and process the results.
"""

import logging
import time
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import requests
from urllib.parse import urlparse, quote_plus

# Initialize logger
logger = logging.getLogger(__name__)


class SearchEngine(Enum):
    """Supported search engines"""
    GOOGLE = "google"
    BING = "bing"
    DUCKDUCKGO = "duckduckgo"
    SERPAPI = "serpapi"


@dataclass
class WebSearchResult:
    """Data class for web search results"""
    title: str
    url: str
    snippet: str
    content: str
    source: str
    search_query: str
    timestamp: str
    language: str = "en"
    word_count: int = 0


class WebSearchClient:
    """
    Client for fetching web search results from various search engines.
    
    This client supports multiple search engines with proper API integration,
    rate limiting, and content extraction.
    """
    
    def __init__(
        self,
        search_engine: SearchEngine = SearchEngine.SERPAPI,
        api_key: Optional[str] = None,
        timeout: int = 30,
        rate_limit_delay: float = 1.0,
        max_results_per_query: int = 10
    ):
        """
        Initialize the web search client.
        
        Args:
            search_engine: Search engine to use
            api_key: API key for the search engine
            timeout: Request timeout in seconds
            rate_limit_delay: Delay between requests to respect rate limits
            max_results_per_query: Maximum results to return per query
        """
        self.search_engine = search_engine
        self.api_key = api_key
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self.max_results_per_query = max_results_per_query
        
        # Validate API key for services that require it
        if search_engine in [SearchEngine.GOOGLE, SearchEngine.SERPAPI] and not api_key:
            logger.warning(f"API key is recommended for {search_engine.value}")
        
        logger.info(f"Web search client initialized for {search_engine.value}")
    
    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        language: str = "en",
        region: str = "us"
    ) -> List[WebSearchResult]:
        """
        Perform a web search and return processed results.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            language: Language code for results
            region: Region code for results
            
        Returns:
            List of WebSearchResult objects
        """
        if max_results is None:
            max_results = self.max_results_per_query
        
        try:
            # Add delay to respect rate limits
            time.sleep(self.rate_limit_delay)
            
            # Perform search based on selected engine
            if self.search_engine == SearchEngine.SERPAPI:
                results = self._search_with_serpapi(query, max_results, language, region)
            elif self.search_engine == SearchEngine.GOOGLE:
                results = self._search_with_google(query, max_results, language, region)
            elif self.search_engine == SearchEngine.BING:
                results = self._search_with_bing(query, max_results, language, region)
            else:
                results = self._search_with_duckduckgo(query, max_results, language, region)
            
            # Process and extract content from results
            processed_results = []
            for result in results[:max_results]:
                processed_result = self._process_result(result, query, language)
                if processed_result:
                    processed_results.append(processed_result)
            
            logger.info(f"Found {len(processed_results)} results for query: {query}")
            return processed_results
            
        except Exception as e:
            logger.error(f"Error searching for '{query}': {e}")
            return []
    
    def _search_with_serpapi(
        self,
        query: str,
        max_results: int,
        language: str,
        region: str
    ) -> List[Dict[str, Any]]:
        """Search using SerpAPI"""
        if not self.api_key:
            raise ValueError("API key is required for SerpAPI")
        
        params = {
            "q": query,
            "api_key": self.api_key,
            "engine": "google",
            "num": max_results,
            "hl": language,
            "gl": region,
            "output": "json"
        }
        
        try:
            response = requests.get(
                "https://serpapi.com/search",
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("organic_results", [])
            
        except requests.RequestException as e:
            logger.error(f"SerpAPI request failed: {e}")
            return []
    
    def _search_with_google(
        self,
        query: str,
        max_results: int,
        language: str,
        region: str
    ) -> List[Dict[str, Any]]:
        """Search using Google Custom Search API"""
        if not self.api_key:
            raise ValueError("API key is required for Google Custom Search")
        
        params = {
            "q": query,
            "key": self.api_key,
            "cx": "YOUR_SEARCH_ENGINE_ID",  # This should be configured
            "num": min(max_results, 10),  # Google limits to 10 results per page
            "hl": language,
            "gl": region
        }
        
        try:
            response = requests.get(
                "https://www.googleapis.com/customsearch/v1",
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("items", [])
            
        except requests.RequestException as e:
            logger.error(f"Google Custom Search request failed: {e}")
            return []
    
    def _search_with_bing(
        self,
        query: str,
        max_results: int,
        language: str,
        region: str
    ) -> List[Dict[str, Any]]:
        """Search using Bing Web Search API"""
        if not self.api_key:
            raise ValueError("API key is required for Bing Web Search")
        
        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key
        }
        
        params = {
            "q": query,
            "count": min(max_results, 50),  # Bing allows up to 50 results
            "mkt": f"{language}-{region}",
            "textFormat": "HTML"
        }
        
        try:
            response = requests.get(
                "https://api.bing.microsoft.com/v7.0/search",
                headers=headers,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("webPages", {}).get("value", [])
            
        except requests.RequestException as e:
            logger.error(f"Bing Web Search request failed: {e}")
            return []
    
    def _search_with_duckduckgo(
        self,
        query: str,
        max_results: int,
        language: str,
        region: str
    ) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo (no API key required)"""
        # DuckDuckGo doesn't have an official API, so we use the HTML endpoint
        # Note: This is a simple implementation and may break if DuckDuckGo changes their HTML
        
        params = {
            "q": query,
            "kl": f"{region}-{language}",
            "ia": "web"
        }
        
        try:
            response = requests.get(
                "https://html.duckduckgo.com/html/",
                params=params,
                timeout=self.timeout,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            )
            response.raise_for_status()
            
            # Parse HTML to extract results (simplified)
            # In a real implementation, you'd use BeautifulSoup or similar
            results = []
            # This is a placeholder - actual implementation would parse HTML
            logger.warning("DuckDuckGo HTML parsing not fully implemented")
            
            return results
            
        except requests.RequestException as e:
            logger.error(f"DuckDuckGo request failed: {e}")
            return []
    
    def _process_result(
        self,
        result: Dict[str, Any],
        query: str,
        language: str
    ) -> Optional[WebSearchResult]:
        """Process a raw search result and extract content"""
        try:
            # Extract basic information
            title = result.get("title", result.get("name", ""))
            url = result.get("link", result.get("url", ""))
            snippet = result.get("snippet", result.get("description", ""))
            
            if not title or not url:
                return None
            
            # Fetch full content from the URL
            content = self._fetch_page_content(url)
            
            # Create search result object
            search_result = WebSearchResult(
                title=title,
                url=url,
                snippet=snippet,
                content=content,
                source=self.search_engine.value,
                search_query=query,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                language=language,
                word_count=len(content.split())
            )
            
            return search_result
            
        except Exception as e:
            logger.warning(f"Failed to process result: {e}")
            return None
    
    def _fetch_page_content(self, url: str) -> str:
        """Fetch and extract main content from a web page"""
        try:
            response = requests.get(
                url,
                timeout=self.timeout,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
            )
            response.raise_for_status()
            
            # For simplicity, return the text content
            # In a real implementation, you'd use BeautifulSoup to extract main content
            # and remove navigation, ads, etc.
            return response.text[:5000]  # Limit content length
            
        except Exception as e:
            logger.warning(f"Failed to fetch content from {url}: {e}")
            return ""
    
    def batch_search(
        self,
        queries: List[str],
        max_results_per_query: Optional[int] = None,
        language: str = "en",
        region: str = "us"
    ) -> Dict[str, List[WebSearchResult]]:
        """
        Perform multiple searches in batch.
        
        Args:
            queries: List of search queries
            max_results_per_query: Maximum results per query
            language: Language code for results
            region: Region code for results
            
        Returns:
            Dictionary mapping queries to search results
        """
        if max_results_per_query is None:
            max_results_per_query = self.max_results_per_query
        
        results = {}
        
        for query in queries:
            try:
                query_results = self.search(
                    query,
                    max_results_per_query,
                    language,
                    region
                )
                results[query] = query_results
                logger.info(f"Completed search for: {query} ({len(query_results)} results)")
            except Exception as e:
                logger.error(f"Failed to search for '{query}': {e}")
                results[query] = []
        
        return results


# Example usage and testing
if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize client (using SerpAPI as example)
    # Note: You need to set SERPAPI_API_KEY environment variable
    import os
    api_key = os.getenv("SERPAPI_API_KEY")
    
    if api_key:
        client = WebSearchClient(
            search_engine=SearchEngine.SERPAPI,
            api_key=api_key
        )
        
        # Test search
        results = client.search("machine learning", max_results=3)
        for result in results:
            print(f"Result: {result.title}")
            print(f"URL: {result.url}")
            print(f"Word count: {result.word_count}")
            print("---")
    else:
        print("SERPAPI_API_KEY not set. Using dummy mode for demonstration.")
        
        # Test with dummy data
        client = WebSearchClient(search_engine=SearchEngine.DUCKDUCKGO)
        results = client.search("artificial intelligence", max_results=2)
        
        if results:
            for result in results:
                print(f"Result: {result.title}")
                print(f"URL: {result.url}")
        else:
            print("No results found (expected without proper API configuration)")