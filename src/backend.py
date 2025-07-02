"""
Core workflow orchestration for the AI Research Assistant
"""
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from models import Analyst, ResearchConfig, ResearchResults
from agents import AnalystGenerator, InterviewAgent, ReportWriter
from services import ServiceManager


logger = logging.getLogger(__name__)


@dataclass
class WorkflowCallbacks:
    """Callbacks for workflow progress updates"""
    on_progress: Optional[Callable[[float, str], None]] = None
    on_analyst_created: Optional[Callable[[List[Analyst]], None]] = None
    on_interview_complete: Optional[Callable[[str, str], None]] = None
    on_section_complete: Optional[Callable[[str], None]] = None
    on_error: Optional[Callable[[str], None]] = None


class ResearchWorkflow:
    """Main workflow orchestrator"""
    
    def __init__(self, service_manager: ServiceManager):
        self.service_manager = service_manager
        self.analyst_generator = AnalystGenerator(service_manager.llm_service)
        self.interview_agent = InterviewAgent(
            service_manager.llm_service,
            service_manager.search_service
        )
        self.report_writer = ReportWriter(service_manager.llm_service)
    
    def run_research(
        self,
        config: ResearchConfig,
        callbacks: Optional[WorkflowCallbacks] = None
    ) -> ResearchResults:
        """
        Run the complete research workflow
        
        Args:
            config: Research configuration
            callbacks: Optional callbacks for progress updates
            
        Returns:
            ResearchResults object containing all outputs
        """
        if callbacks is None:
            callbacks = WorkflowCallbacks()
        
        try:
            # Step 1: Generate analysts (10-25% progress)
            self._update_progress(callbacks, 10, "Generating research analysts...")
            analysts = self._generate_analysts(config, callbacks)
            self._update_progress(callbacks, 25, f"Generated {len(analysts)} analysts")
            
            # Step 2: Conduct interviews (25-70% progress)
            self._update_progress(callbacks, 30, "Starting research interviews...")
            sections = self._conduct_interviews(
                analysts, config.topic, config.max_turns, callbacks
            )
            self._update_progress(callbacks, 70, "All interviews completed")
            
            # Step 3: Generate final report (70-100% progress)
            self._update_progress(callbacks, 85, "Generating final report...")
            final_report = self._generate_report(sections, config.topic, callbacks)
            self._update_progress(callbacks, 100, "Research completed successfully!")
            
            # Create results
            results = ResearchResults(
                topic=config.topic,
                analysts=analysts,
                sections=sections,
                final_report=final_report
            )
            
            return results
            
        except Exception as e:
            error_msg = f"Workflow failed: {str(e)}"
            logger.error(error_msg)
            if callbacks.on_error:
                callbacks.on_error(error_msg)
            raise
    
    def _generate_analysts(
        self,
        config: ResearchConfig,
        callbacks: WorkflowCallbacks
    ) -> List[Analyst]:
        """Generate analyst personas"""
        analysts = self.analyst_generator.generate_analysts(
            topic=config.topic,
            max_analysts=config.max_analysts,
            human_feedback=config.human_feedback
        )
        
        if callbacks.on_analyst_created:
            callbacks.on_analyst_created(analysts)
        
        return analysts
    
    def _conduct_interviews(
        self,
        analysts: List[Analyst],
        topic: str,
        max_turns: int,
        callbacks: WorkflowCallbacks
    ) -> List[str]:
        """Conduct interviews with all analysts"""
        sections = []
        total_analysts = len(analysts)
        
        for i, analyst in enumerate(analysts):
            try:
                # Update progress
                progress = 30 + (i * 40 // total_analysts)
                self._update_progress(
                    callbacks,
                    progress,
                    f"Interviewing {analyst.name}... ({i+1}/{total_analysts})"
                )
                
                # Conduct interview
                section = self.interview_agent.conduct_interview(
                    analyst=analyst,
                    topic=topic,
                    max_turns=max_turns
                )
                sections.append(section)
                
                # Notify completion
                if callbacks.on_interview_complete:
                    callbacks.on_interview_complete(analyst.name, section)
                
                if callbacks.on_section_complete:
                    callbacks.on_section_complete(section)
                    
            except Exception as e:
                error_msg = f"Interview failed for {analyst.name}: {str(e)}"
                logger.error(error_msg)
                sections.append(f"## Error\nInterview with {analyst.name} failed: {str(e)}")
        
        return sections
    
    def _generate_report(
        self,
        sections: List[str],
        topic: str,
        callbacks: WorkflowCallbacks
    ) -> str:
        """Generate the final report"""
        final_report = self.report_writer.write_report(sections, topic)
        return final_report
    
    def _update_progress(
        self,
        callbacks: WorkflowCallbacks,
        progress: float,
        message: str
    ):
        """Update progress through callback"""
        if callbacks.on_progress:
            callbacks.on_progress(progress, message)


class ParallelResearchWorkflow(ResearchWorkflow):
    """Research workflow with parallel interview execution"""
    
    def __init__(self, service_manager: ServiceManager, max_workers: int = 3):
        super().__init__(service_manager)
        self.max_workers = max_workers
    
    def _conduct_interviews(
        self,
        analysts: List[Analyst],
        topic: str,
        max_turns: int,
        callbacks: WorkflowCallbacks
    ) -> List[str]:
        """Conduct interviews in parallel"""
        sections = [None] * len(analysts)
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all interview tasks
            future_to_index = {
                executor.submit(
                    self.interview_agent.conduct_interview,
                    analyst=analyst,
                    topic=topic,
                    max_turns=max_turns
                ): i
                for i, analyst in enumerate(analysts)
            }
            
            # Process completed interviews
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                analyst = analysts[index]
                
                try:
                    section = future.result()
                    sections[index] = section
                    completed += 1
                    
                    # Update progress
                    progress = 30 + (completed * 40 // len(analysts))
                    self._update_progress(
                        callbacks,
                        progress,
                        f"Completed interview with {analyst.name} ({completed}/{len(analysts)})"
                    )
                    
                    # Notify completion
                    if callbacks.on_interview_complete:
                        callbacks.on_interview_complete(analyst.name, section)
                    
                except Exception as e:
                    error_msg = f"Interview failed for {analyst.name}: {str(e)}"
                    logger.error(error_msg)
                    sections[index] = f"## Error\nInterview with {analyst.name} failed: {str(e)}"
                    completed += 1
        
        # Filter out None values
        return [s for s in sections if s is not None]


class WorkflowFactory:
    """Factory for creating workflow instances"""
    
    @staticmethod
    def create_workflow(
        service_manager: ServiceManager,
        parallel: bool = False,
        max_workers: int = 3
    ) -> ResearchWorkflow:
        """
        Create a workflow instance
        
        Args:
            service_manager: Service manager instance
            parallel: Whether to use parallel processing
            max_workers: Maximum workers for parallel processing
            
        Returns:
            ResearchWorkflow instance
        """
        if parallel:
            return ParallelResearchWorkflow(service_manager, max_workers)
        else:
            return ResearchWorkflow(service_manager)


# Convenience functions for standalone usage
def run_research(
    topic: str,
    max_analysts: int = 3,
    max_turns: int = 2,
    human_feedback: str = "",
    openai_api_key: Optional[str] = None,
    tavily_api_key: Optional[str] = None,
    parallel: bool = False,
    callbacks: Optional[WorkflowCallbacks] = None
) -> ResearchResults:
    """
    Convenience function to run research with minimal setup
    
    Args:
        topic: Research topic
        max_analysts: Number of analysts to create
        max_turns: Maximum interview turns
        human_feedback: Optional human feedback for analyst creation
        openai_api_key: OpenAI API key (uses env if not provided)
        tavily_api_key: Tavily API key (uses env if not provided)
        parallel: Whether to use parallel processing
        callbacks: Optional callbacks for progress updates
        
    Returns:
        ResearchResults object
    """
    # Create service manager
    service_manager = ServiceManager(
        openai_api_key=openai_api_key,
        tavily_api_key=tavily_api_key
    )
    
    # Create config
    config = ResearchConfig(
        topic=topic,
        max_analysts=max_analysts,
        max_turns=max_turns,
        human_feedback=human_feedback
    )
    
    # Create and run workflow
    workflow = WorkflowFactory.create_workflow(service_manager, parallel=parallel)
    return workflow.run_research(config, callbacks)