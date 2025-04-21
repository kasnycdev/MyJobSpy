
**2. Improve/Augment Scraped Data & Add User Context (Code)**

**(a) Update Pydantic Models (`models.py` or similar)**

```python
# models.py
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
import re

# Model for the original scraped job data (subset of JobSpy fields + our additions)
class OriginalJobData(BaseModel):
    id: Optional[str] = None
    site: Optional[str] = None
    url: Optional[str] = None
    job_url_direct: Optional[str] = None
    title: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    date_posted: Optional[str] = None # Will be normalized later if needed
    employment_type: Optional[str] = None
    salary_source: Optional[str] = None
    interval: Optional[str] = None
    min_amount: Optional[float] = None
    max_amount: Optional[float] = None
    currency: Optional[str] = None
    is_remote: Optional[bool] = None
    job_level: Optional[str] = None
    job_function: Optional[str] = None
    listing_type: Optional[str] = None
    emails: Optional[str | List[str]] = None # JobSpy might return str or list
    description: Optional[str] = None
    company_industry: Optional[str] = None
    company_url: Optional[str] = None
    company_logo: Optional[str] = None
    company_url_direct: Optional[str] = None
    company_addresses: Optional[str] = None
    company_num_employees: Optional[str | int] = None # Can vary
    company_revenue: Optional[str] = None
    company_description: Optional[str] = None
    skills: Optional[str | List[str]] = None
    experience_range: Optional[str] = None
    company_rating: Optional[float] = None
    company_reviews_count: Optional[int] = None
    vacancy_count: Optional[int] = None
    work_from_home_type: Optional[str] = None
    salary_text: Optional[str] = None # Keep raw text for potential parsing
    benefits_text: Optional[str] = None

    # --- Our Added Fields for Context/Processing ---
    salary_estimated: bool = False
    normalized_date_posted: Optional[datetime] = None


# Model for Keyword Analysis (nested in JobAnalysisResult)
class KeywordAnalysis(BaseModel):
    matched_required: Optional[List[str]] = []
    missing_required: Optional[List[str]] = []
    missing_preferred: Optional[List[str]] = []

# Model for Match Details (nested in JobAnalysisResult)
class MatchDetails(BaseModel):
    assessment: Optional[Literal["Strong Match", "Partial Match", "Weak Match", "Unknown"]] = "Unknown"
    reasoning: Optional[str] = None

# Model for the AI Analysis Result
class JobAnalysisResult(BaseModel):
    suitability_score: Optional[int] = Field(None, ge=0, le=100)
    justification: Optional[str] = None
    keyword_analysis: Optional[KeywordAnalysis] = KeywordAnalysis() # Default to empty lists
    skill_match_details: Optional[MatchDetails] = MatchDetails()
    experience_match_details: Optional[MatchDetails] = MatchDetails()
    qualification_match_details: Optional[MatchDetails] = MatchDetails()
    salary_alignment: Optional[Literal[
        "Below User Range", "Within User Range", "Above User Range",
        "Partially Overlaps", "Unknown"
    ]] = "Unknown"
    alignment_details: Optional[str] = None
    date_analyzed: datetime = Field(default_factory=datetime.now)

# Top-level model for the combined results in the JSON file
class CombinedJobResult(BaseModel):
    original_job_data: OriginalJobData
    analysis: Optional[JobAnalysisResult] = None # Analysis might fail or be skipped