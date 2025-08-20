"""
Data Acquisition Module for PrismRAG

This module provides functionality to acquire real data from various sources
including Wikipedia, web search results, and knowledge graphs.
"""

from .wikipedia_client import WikipediaClient
from .web_search_client import WebSearchClient
from .knowledge_graph_client import KnowledgeGraphClient
from .data_fetcher import DataFetcher

__all__ = [
    'WikipediaClient',
    'WebSearchClient',
    'KnowledgeGraphClient',
    'DataFetcher'
]