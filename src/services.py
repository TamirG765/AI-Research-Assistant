"""
External service integrations for the AI Research Assistant
"""
import logging
from typing import Optional, List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

# Import Tavily with fallback
try:
    from langchain_tavily import TavilySearchResults
except ImportError:
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
    except ImportError:
        TavilySearchResults = None

from config import settings


logger = logging.getLogger(__name__)


class LLMService:
    """Service for managing LLM interactions"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.openai_api_key
        self._llm = None
    
    @property
    def llm(self) -> ChatOpenAI:
        """Lazy initialization of LLM"""
        if self._llm is None:
            if not self.api_key:
                raise ValueError("OpenAI API key not provided")
            
            self._llm = ChatOpenAI(
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                api_key=self.api_key
            )
            logger.info(f"Initialized LLM with model: {settings.llm_model}")
        
        return self._llm
    
    def get_structured_llm(self, output_class):
        """Get LLM with structured output"""
        return self.llm.with_structured_output(output_class)
    
    def invoke(self, messages: List[Any]) -> Any:
        """Invoke LLM with error handling"""
        try:
            return self.llm.invoke(messages)
        except Exception as e:
            logger.error(f"LLM invocation failed: {str(e)}")
            raise


class SearchService:
    """Service for managing web search operations"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.tavily_api_key
        self._search_tool = None
    
    @property
    def search_tool(self) -> TavilySearchResults:
        """Lazy initialization of search tool"""
        if self._search_tool is None:
            if TavilySearchResults is None:
                raise ImportError("Tavily search not available. Please install langchain-tavily")
            
            if not self.api_key:
                raise ValueError("Tavily API key not provided")
            
            self._search_tool = TavilySearchResults(
                max_results=settings.max_search_results,
                tavily_api_key=self.api_key
            )
            logger.info("Initialized Tavily search tool")
        
        return self._search_tool
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """Execute search with error handling"""
        try:
            logger.info(f"Searching for: {query}")
            results = self.search_tool.invoke(query)
            logger.info(f"Found {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {str(e)}")
            return []
    
    def format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results as documents"""
        if not results:
            return ""
        
        formatted_docs = []
        for doc in results:
            formatted_doc = f'<Document href="{doc.get("url", "")}"/>\n{doc.get("content", "")}\n</Document>'
            formatted_docs.append(formatted_doc)
        
        return "\n\n---\n\n".join(formatted_docs)


class ServiceManager:
    """Manager for all external services"""
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        tavily_api_key: Optional[str] = None
    ):
        self.llm_service = LLMService(openai_api_key)
        self.search_service = SearchService(tavily_api_key)
    
    @classmethod
    def from_env(cls) -> "ServiceManager":
        """Create ServiceManager from environment variables"""
        return cls()
    
    def validate_services(self) -> Dict[str, bool]:
        """Validate all services are properly configured"""
        validation = {
            "llm": False,
            "search": False
        }
        
        try:
            # Test LLM
            self.llm_service.llm
            validation["llm"] = True
        except Exception as e:
            logger.error(f"LLM validation failed: {str(e)}")
        
        try:
            # Test Search
            self.search_service.search_tool
            validation["search"] = True
        except Exception as e:
            logger.error(f"Search validation failed: {str(e)}")
        
        return validation