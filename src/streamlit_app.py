"""
Streamlit UI for the AI Research Assistant
"""
import streamlit as st
import time
import logging
from typing import List

from models import ResearchConfig, Analyst
from backend import WorkflowFactory, WorkflowCallbacks
from services import ServiceManager
from utils import (
    StreamlitAPIKeyManager,
    ProgressTracker,
    display_analyst_card,
    create_download_button,
    sanitize_filename,
    setup_logging
)
from config import settings


# Setup logging
setup_logging(settings.log_level)
logger = logging.getLogger(__name__)


# Streamlit page config
st.set_page_config(
    page_title=settings.app_title,
    page_icon=settings.app_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .progress-container {
        margin: 2rem 0;
    }
    .analyst-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f8f9fa;
        color: #212529;
    }
    .analyst-card strong {
        color: #1f77b4;
    }
    .analyst-card em {
        color: #6c757d;
    }
    .step-indicator {
        font-weight: bold;
        color: #28a745;
        margin: 1rem 0;
    }
    .error-message {
        color: #dc3545;
        padding: 1rem;
        border: 1px solid #dc3545;
        border-radius: 5px;
        background-color: #f8d7da;
    }
    .success-message {
        color: #155724;
        padding: 1rem;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        background-color: #d4edda;
    }
</style>
""", unsafe_allow_html=True)


class ResearchAssistantApp:
    """Main Streamlit application class"""
    
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'workflow_completed' not in st.session_state:
            st.session_state.workflow_completed = False
        if 'analysts' not in st.session_state:
            st.session_state.analysts = None
        if 'final_report' not in st.session_state:
            st.session_state.final_report = None
        if 'research_results' not in st.session_state:
            st.session_state.research_results = None
        if 'error_message' not in st.session_state:
            st.session_state.error_message = None
    
    def render_header(self):
        """Render application header"""
        st.markdown(
            f'<div class="main-header">{settings.app_icon} {settings.app_title}</div>',
            unsafe_allow_html=True
        )
        st.markdown("**Automated multi-agent research system powered by LangGraph**")
    
    def render_sidebar(self) -> dict:
        """Render sidebar configuration and return config"""
        with st.sidebar:
            # Get API keys from environment/secrets only
            openai_key = StreamlitAPIKeyManager.get_api_key(
                "OpenAI API Key",
                "OPENAI_API_KEY",
                sidebar=False  # Don't show in sidebar
            )
            tavily_key = StreamlitAPIKeyManager.get_api_key(
                "Tavily API Key",
                "TAVILY_API_KEY",
                sidebar=False  # Don't show in sidebar
            )
            
            # Research Configuration
            st.header("🔬 Research Settings")
            
            topic = st.text_area(
                "Research Topic",
                value="The benefits of adopting LangGraph as an agent framework",
                height=100,
                help="Enter the topic you want to research"
            )
            
            max_analysts = st.slider(
                "Number of Analysts",
                min_value=2,
                max_value=5,
                value=settings.default_max_analysts,
                help="More analysts = more comprehensive research but longer processing time"
            )
            
            max_turns = st.slider(
                "Interview Turns per Analyst",
                min_value=1,
                max_value=3,
                value=settings.default_max_turns,
                help="Number of question-answer exchanges per interview"
            )
            
            st.divider()
            
            # Advanced Options
            with st.expander("⚡ Advanced Options"):
                use_parallel = st.checkbox(
                    "Use Parallel Processing",
                    value=True,
                    help="Conduct interviews in parallel for faster results"
                )
                
                if use_parallel:
                    max_workers = st.slider(
                        "Max Parallel Workers",
                        min_value=1,
                        max_value=5,
                        value=3,
                        help="Number of concurrent interviews"
                    )
                else:
                    max_workers = 1
                
                use_feedback = st.checkbox("Add custom analyst perspective")
                human_feedback = ""
                if use_feedback:
                    human_feedback = st.text_area(
                        "Additional Perspective Request",
                        placeholder="e.g., Add someone from a startup perspective",
                        help="Request specific analyst perspectives to be included"
                    )
            
            st.divider()
            
            # Action Buttons
            start_research = st.button(
                "🚀 Start Research",
                type="primary",
                use_container_width=True,
                disabled=not (openai_key and tavily_key)
            )
            
            return {
                "topic": topic,
                "max_analysts": max_analysts,
                "max_turns": max_turns,
                "human_feedback": human_feedback,
                "openai_key": openai_key,
                "tavily_key": tavily_key,
                "use_parallel": use_parallel,
                "max_workers": max_workers,
                "start_research": start_research
            }
    
    def create_workflow_callbacks(self, progress_tracker: ProgressTracker) -> WorkflowCallbacks:
        """Create callbacks for workflow progress updates"""
        
        def on_progress(progress: float, message: str):
            progress_tracker.update(progress, message)
        
        def on_analyst_created(analysts: List[Analyst]):
            st.session_state.analysts = analysts
        
        def on_error(error_message: str):
            st.session_state.error_message = error_message
        
        return WorkflowCallbacks(
            on_progress=on_progress,
            on_analyst_created=on_analyst_created,
            on_error=on_error
        )
    
    def run_research_workflow(self, config: dict):
        """Run the research workflow"""
        # Create progress tracking UI
        progress_bar = st.progress(0)
        status_text = st.empty()
        progress_tracker = ProgressTracker(progress_bar, status_text)
        
        try:
            # Create service manager
            service_manager = ServiceManager(
                openai_api_key=config["openai_key"],
                tavily_api_key=config["tavily_key"]
            )
            
            # Validate services
            validation = service_manager.validate_services()
            if not all(validation.values()):
                failed_services = [k for k, v in validation.items() if not v]
                raise ValueError(f"Failed to initialize services: {', '.join(failed_services)}")
            
            # Create research config
            research_config = ResearchConfig(
                topic=config["topic"],
                max_analysts=config["max_analysts"],
                max_turns=config["max_turns"],
                human_feedback=config["human_feedback"]
            )
            
            # Create workflow
            workflow = WorkflowFactory.create_workflow(
                service_manager,
                parallel=config["use_parallel"],
                max_workers=config["max_workers"]
            )
            
            # Create callbacks
            callbacks = self.create_workflow_callbacks(progress_tracker)
            
            # Run workflow
            results = workflow.run_research(research_config, callbacks)
            
            # Store results
            st.session_state.research_results = results
            st.session_state.final_report = results.final_report
            st.session_state.workflow_completed = True
            
            # Show success message
            progress_tracker.update(100, "✅ Research completed successfully!")
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Research workflow failed: {str(e)}")
            st.error(f"❌ Research failed: {str(e)}")
            st.session_state.error_message = str(e)
    
    def display_results(self):
        """Display research results"""
        if st.session_state.analysts:
            with st.expander("👥 Generated Research Analysts", expanded=True):
                for analyst in st.session_state.analysts:
                    display_analyst_card(analyst)
        
        if st.session_state.final_report:
            st.markdown("---")
            st.subheader("📊 Final Research Report")
            st.markdown(st.session_state.final_report)
            
            # Download button
            topic = st.session_state.research_results.topic if st.session_state.research_results else "research"
            filename = f"research_report_{sanitize_filename(topic)}.md"
            create_download_button(
                st.session_state.final_report,
                filename,
                "📥 Download Report"
            )
    
    def render_info_section(self):
        """Render information section"""
        with st.expander("ℹ️ About This System"):
            st.markdown("""
            ### Features
            - **Multi-Agent System**: Multiple AI analysts with diverse perspectives
            - **Intelligent Interviews**: Dynamic Q&A sessions with web search integration
            - **Parallel Processing**: Optional concurrent interview execution
            - **Comprehensive Reports**: Automated synthesis with proper citations
            - **Modular Architecture**: Clean separation of UI, workflow, and services
            
            ### How It Works
            1. **Analyst Generation**: Creates diverse expert personas based on your topic
            2. **Research Interviews**: Conducts structured interviews with web search
            3. **Report Synthesis**: Compiles findings into a comprehensive report
            
            ### Technologies
            - **LangGraph**: Workflow orchestration
            - **LangChain**: LLM integration
            - **Streamlit**: Interactive UI
            - **OpenAI GPT-4**: Language model
            - **Tavily**: Web search API
            """)
    
    def run(self):
        """Main application entry point"""
        self.render_header()
        
        # Get configuration from sidebar
        config = self.render_sidebar()
        
        # Check if research should start
        if config["start_research"] and config["topic"]:
            if not config["openai_key"] or not config["tavily_key"]:
                st.error("❌ Please provide both API keys to start research")
            else:
                self.run_research_workflow(config)
        
        # Display results
        if st.session_state.workflow_completed:
            self.display_results()
        elif st.session_state.error_message:
            st.markdown(
                f'<div class="error-message">❌ {st.session_state.error_message}</div>',
                unsafe_allow_html=True
            )
        
        # Information section
        self.render_info_section()


def main():
    """Main function to run the Streamlit app"""
    app = ResearchAssistantApp()
    app.run()


if __name__ == "__main__":
    main()