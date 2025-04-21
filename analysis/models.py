from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any

# --- Model Definitions ---

class ExperienceItem(BaseModel):
    title: Optional[str] = Field(None, description="Job title")
    company: Optional[str] = Field(None, description="Company name")
    years: Optional[Union[float, str]] = Field(None, description="Years in the role")
    description: Optional[str] = Field(None, description="Full description of responsibilities/achievements")

class EducationItem(BaseModel):
    degree: Optional[str] = Field(None, description="Degree obtained")
    institution: Optional[str] = Field(None, description="Institution name")
    years: Optional[str] = Field(None, description="Years attended or graduation year")

class AccomplishmentItem(BaseModel):
    title: Optional[str] = Field(None, description="Title/heading of the accomplishment block")
    description: Optional[str] = Field(None, description="Full description of the accomplishment")

class ResumeData(BaseModel):
    """Structured data extracted from a resume."""
    summary: Optional[str] = Field(None, description="Full professional summary")
    management_skills: List[str] = Field([], description="List of management skills")
    technical_skills: List[str] = Field([], description="List of technical skills")
    key_accomplishments: List[AccomplishmentItem] = Field([], description="List of key accomplishments")
    experience: List[ExperienceItem] = Field([], description="Work experience history")
    education: List[EducationItem] = Field([], description="Educational background")
    total_years_experience: Optional[Union[float, str]] = Field(None, description="Estimated total years of professional experience (float preferred)")

class JobAnalysisResult(BaseModel):
    """Analysis results for a single job compared to the resume."""
    suitability_score: int = Field(..., ge=0, le=100, description="Overall suitability score (0-100%)")
    justification: str = Field(..., description="Textual explanation for the score")
    skill_match: Optional[bool] = Field(None, description="Does the resume have the core required skills?")
    experience_match: Optional[bool] = Field(None, description="Does the resume meet the required experience level?")
    qualification_match: Optional[bool] = Field(None, description="Does the resume meet the core qualifications?")
    salary_alignment: Optional[str] = Field(None, description="Assessment of salary fit (e.g., 'Likely Fit', 'Below Range', 'Above Range', 'N/A')")
    benefit_alignment: Optional[str] = Field(None, description="Brief assessment of benefit alignment (e.g., 'Mentions Health', 'N/A')")
    missing_keywords: List[str] = Field([], description="Key required skills/keywords potentially missing from the resume")

class AnalyzedJob(BaseModel):
    """Combines original job data with analysis results."""
    original_job_data: Dict[str, Any] = Field(..., description="The original job mandate dictionary")
    analysis: JobAnalysisResult = Field(..., description="The GenAI analysis results")