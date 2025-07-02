"""
Agent implementations for the AI Research Assistant
"""
import logging
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.messages import get_buffer_string

from models import Analyst, Perspectives, SearchQuery
from config import PromptTemplates
from services import LLMService, SearchService


logger = logging.getLogger(__name__)


class AnalystGenerator:
    """Agent for generating research analysts"""
    
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
    
    def generate_analysts(
        self,
        topic: str,
        max_analysts: int,
        human_feedback: str = ""
    ) -> List[Analyst]:
        """Generate a set of analyst personas"""
        logger.info(f"Generating {max_analysts} analysts for topic: {topic}")
        
        # Format the prompt
        prompt = PromptTemplates.ANALYST_CREATION.format(
            topic=topic,
            max_analysts=max_analysts,
            human_analyst_feedback=human_feedback
        )
        
        # Get structured LLM
        structured_llm = self.llm_service.get_structured_llm(Perspectives)
        
        # Generate analysts
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content="Generate the set of analysts.")
        ]
        
        try:
            perspectives = structured_llm.invoke(messages)
            logger.info(f"Successfully generated {len(perspectives.analysts)} analysts")
            return perspectives.analysts
        except Exception as e:
            logger.error(f"Failed to generate analysts: {str(e)}")
            raise


class InterviewAgent:
    """Agent for conducting interviews"""
    
    def __init__(self, llm_service: LLMService, search_service: SearchService):
        self.llm_service = llm_service
        self.search_service = search_service
    
    def conduct_interview(
        self,
        analyst: Analyst,
        topic: str,
        max_turns: int = 2
    ) -> str:
        """Conduct an interview and return the section content"""
        logger.info(f"Starting interview with {analyst.name}")
        
        messages = [HumanMessage(content=f"So you said you were writing an article on {topic}?")]
        context = []
        
        for turn in range(max_turns):
            # Generate question
            question = self._generate_question(analyst, messages)
            messages.append(question)
            
            # Check if interview is complete
            if "Thank you so much for your help" in question.content:
                logger.info(f"Interview completed early at turn {turn + 1}")
                break
            
            # Generate search query
            search_query = self._generate_search_query(messages)
            
            # Search for information
            search_results = self.search_service.search(search_query)
            formatted_results = self.search_service.format_search_results(search_results)
            
            if formatted_results:
                context.append(formatted_results)
                
                # Generate answer
                answer = self._generate_answer(analyst, messages, formatted_results)
                messages.append(answer)
            else:
                logger.warning(f"No search results for query: {search_query}")
        
        # Write section
        section = self._write_section(analyst, context)
        logger.info(f"Interview with {analyst.name} completed")
        
        return section
    
    def _generate_question(self, analyst: Analyst, messages: List[Any]) -> AIMessage:
        """Generate interview question"""
        prompt = PromptTemplates.INTERVIEW_QUESTION.format(
            analyst_persona=analyst.persona
        )
        
        question = self.llm_service.invoke([SystemMessage(content=prompt)] + messages)
        question.name = "analyst"
        return question
    
    def _generate_search_query(self, messages: List[Any]) -> str:
        """Generate search query from conversation"""
        structured_llm = self.llm_service.get_structured_llm(SearchQuery)
        
        search_instructions = SystemMessage(content=PromptTemplates.SEARCH_QUERY_GENERATION)
        search_query = structured_llm.invoke([search_instructions] + messages)
        
        return search_query.search_query
    
    def _generate_answer(
        self,
        analyst: Analyst,
        messages: List[Any],
        context: str
    ) -> AIMessage:
        """Generate expert answer"""
        prompt = PromptTemplates.EXPERT_ANSWER.format(
            goals=analyst.persona,
            context=context
        )
        
        answer = self.llm_service.invoke([SystemMessage(content=prompt)] + messages)
        answer.name = "expert"
        return answer
    
    def _write_section(self, analyst: Analyst, context: List[str]) -> str:
        """Write report section"""
        all_context = "\n\n".join(context)
        
        prompt = PromptTemplates.SECTION_WRITER.format(
            analyst_description=analyst.description
        )
        
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=f"Use this source to write your section: {all_context}")
        ]
        
        section = self.llm_service.invoke(messages)
        return section.content


class ReportWriter:
    """Agent for writing the final report"""
    
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
    
    def write_report(self, sections: List[str], topic: str) -> str:
        """Generate the final report"""
        logger.info("Starting final report generation")
        
        # Generate main content
        content = self._generate_content(sections, topic)
        
        # Generate introduction
        introduction = self._generate_intro_conclusion(sections, topic, "introduction")
        
        # Generate conclusion
        conclusion = self._generate_intro_conclusion(sections, topic, "conclusion")
        
        # Compile final report
        final_report = self._compile_report(introduction, content, conclusion)
        
        logger.info("Final report generation completed")
        return final_report
    
    def _generate_content(self, sections: List[str], topic: str) -> str:
        """Generate main report content"""
        sections_text = chr(10).join(sections)
        prompt = PromptTemplates.FINAL_REPORT_WRITER.format(
            topic=topic,
            sections=sections_text
        )
        
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content="Write a report based upon these memos.")
        ]
        
        content = self.llm_service.invoke(messages)
        return content.content
    
    def _generate_intro_conclusion(
        self,
        sections: List[str],
        topic: str,
        section_type: str
    ) -> str:
        """Generate introduction or conclusion"""
        sections_text = chr(10).join(sections)
        prompt = PromptTemplates.INTRO_CONCLUSION_WRITER.format(
            topic=topic,
            sections=sections_text
        )
        
        instruction = (
            "Write the report introduction"
            if section_type == "introduction"
            else "Write the report conclusion"
        )
        
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=instruction)
        ]
        
        result = self.llm_service.invoke(messages)
        return result.content
    
    def _compile_report(
        self,
        introduction: str,
        content: str,
        conclusion: str
    ) -> str:
        """Compile the final report"""
        # Clean up content
        if content.startswith("## Insights"):
            content = content.replace("## Insights", "").strip()
        
        # Extract sources if present
        sources = None
        if "## Sources" in content:
            try:
                content, sources = content.split("\n## Sources\n", 1)
            except:
                sources = None
        
        # Compile report
        final_report = introduction + "\n\n---\n\n" + content + "\n\n---\n\n" + conclusion
        
        # Add sources if available
        if sources is not None:
            final_report += "\n\n## Sources\n" + sources
        
        return final_report