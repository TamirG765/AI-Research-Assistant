"""
Utility functions for the AI Research Assistant
"""
import logging
import time
from functools import wraps
from typing import Any, Callable, Optional
import streamlit as st


logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def timer(func: Callable) -> Callable:
    """Decorator to time function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper


def retry_with_backoff(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """Decorator for retrying functions with exponential backoff"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_count = 0
            delay = 1.0
            
            while retry_count < max_retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        logger.error(f"{func.__name__} failed after {max_retries} retries")
                        raise
                    
                    logger.warning(
                        f"{func.__name__} failed (attempt {retry_count}/{max_retries}): {str(e)}. "
                        f"Retrying in {delay} seconds..."
                    )
                    time.sleep(delay)
                    delay *= backoff_factor
            
        return wrapper
    return decorator


class StreamlitAPIKeyManager:
    """Manager for handling API keys in Streamlit"""
    
    @staticmethod
    def get_api_key(
        key_name: str,
        env_name: str,
        sidebar: bool = True
    ) -> Optional[str]:
        """
        Get API key from multiple sources
        
        Args:
            key_name: Display name for the key
            env_name: Environment variable name
            sidebar: Whether to show input in sidebar
            
        Returns:
            API key if found, None otherwise
        """
        api_key = None
        
        # Check Streamlit secrets (with proper error handling)
        try:
            if hasattr(st, 'secrets') and env_name in st.secrets:
                api_key = st.secrets[env_name]
                return api_key
        except Exception:
            # Secrets not available or not found
            pass
        
        # Check environment variables
        import os
        if os.getenv(env_name):
            api_key = os.getenv(env_name)
            return api_key
        
        # Ask user for input
        container = st.sidebar if sidebar else st
        with container:
            st.warning(f"{key_name} not found in secrets or environment")
            api_key = st.text_input(
                f"Enter {key_name}:",
                type="password",
                key=f"api_key_input_{env_name}"
            )
        
        return api_key if api_key else None


class ProgressTracker:
    """Helper class for tracking progress in Streamlit"""
    
    def __init__(self, progress_bar=None, status_text=None):
        self.progress_bar = progress_bar
        self.status_text = status_text
    
    def update(self, progress: float, message: str):
        """Update progress and status"""
        if self.progress_bar:
            self.progress_bar.progress(progress / 100)
        if self.status_text:
            self.status_text.text(message)


def format_markdown_report(report: str) -> str:
    """Format markdown report for better display"""
    # Add any specific formatting if needed
    return report


def create_download_button(
    content: str,
    filename: str,
    label: str = "📥 Download",
    mime_type: str = "text/markdown"
) -> None:
    """Create a download button for content"""
    st.download_button(
        label=label,
        data=content,
        file_name=filename,
        mime=mime_type,
        use_container_width=True
    )


def display_analyst_card(analyst: Any) -> None:
    """Display an analyst card in Streamlit"""
    st.markdown(f"""
    <div class="analyst-card">
        <strong>{analyst.name}</strong> - {analyst.role}<br>
        <em>{analyst.affiliation}</em><br>
        <small>{analyst.description}</small>
    </div>
    """, unsafe_allow_html=True)


def sanitize_filename(filename: str, max_length: int = 50) -> str:
    """Sanitize filename for safe file operations"""
    # Replace spaces and special characters
    safe_name = filename.replace(' ', '_')
    safe_name = ''.join(c for c in safe_name if c.isalnum() or c in ('_', '-'))
    
    # Truncate if too long
    if len(safe_name) > max_length:
        safe_name = safe_name[:max_length]
    
    return safe_name