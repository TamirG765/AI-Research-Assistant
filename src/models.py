"""
Data models for the AI Research Assistant
"""
from typing import List
from pydantic import BaseModel, Field


class Analyst(BaseModel):
    """Represents an AI analyst persona"""
    affiliation: str = Field(description="Primary affiliation of the analyst.")
    name: str = Field(description="Name of the analyst.")
    role: str = Field(description="Role of the analyst in the context of the topic.")
    description: str = Field(description="Description of the analyst focus, concerns, and motives.")
    
    @property
    def persona(self) -> str:
        """Generate the persona description for the analyst"""
        return (
            f"Name: {self.name}\n"
            f"Role: {self.role}\n"
            f"Affiliation: {self.affiliation}\n"
            f"Description: {self.description}\n"
        )


class Perspectives(BaseModel):
    """Collection of analyst perspectives"""
    analysts: List[Analyst] = Field(
        description="Comprehensive list of analysts with their roles and affiliations."
    )


class SearchQuery(BaseModel):
    """Search query for information retrieval"""
    search_query: str = Field(None, description="Search query for retrieval.")


class ResearchConfig(BaseModel):
    """Configuration for research workflow"""
    topic: str = Field(description="Research topic")
    max_analysts: int = Field(default=3, description="Maximum number of analysts")
    max_turns: int = Field(default=2, description="Maximum interview turns")
    human_feedback: str = Field(default="", description="Human analyst feedback")


class ResearchResults(BaseModel):
    """Results from the research workflow"""
    topic: str
    analysts: List[Analyst]
    sections: List[str]
    final_report: str
    introduction: str = ""
    content: str = ""
    conclusion: str = ""