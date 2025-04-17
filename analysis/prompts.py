# Prompts for the LLM

# NOTE: These prompts are examples. You MUST refine them based on the specific LLM
# you use (e.g., Llama 3, Mistral, Phi-3) and the desired level of detail.
# Pay close attention to formatting instructions for the LLM.

RESUME_EXTRACTION_PROMPT_TEMPLATE = """
You are an expert HR assistant tasked with extracting structured information from a resume text.
Analyze the following resume text and extract the key information requested below.
Output *only* a valid JSON object containing the extracted data, matching the specified structure. Do not include any introductory text, explanations, or markdown formatting.

Resume Text:
---
{resume_text}
---

JSON Structure to populate:
{{
  "summary": "string (Brief professional summary, max 3 sentences)",
  "skills": ["string (List of key technical and soft skills mentioned)"],
  "experience": [
    {{
      "title": "string (Job title)",
      "company": "string (Company name)",
      "years": "float or string (Approximate years in role, e.g., 2.5 or '3 years')",
      "description": "string (Brief summary of responsibilities/achievements)"
    }}
  ],
  "education": [
    {{
      "degree": "string (Degree name)",
      "institution": "string (Institution name)",
      "years": "string (Years attended or graduation year)"
    }}
  ],
  "total_years_experience": "float or string (Estimate total years of relevant professional experience based on the text)"
}}

Extracted JSON:
"""


SUITABILITY_ANALYSIS_PROMPT_TEMPLATE = """
You are an expert HR analyst comparing a candidate's resume against a job description.
Analyze the provided structured resume data and the job description text.
Provide a detailed suitability analysis, focusing on skills, experience, qualifications, salary (if available), and benefits (if mentioned).

Candidate Resume Data (JSON):
---
{resume_data_json}
---

Job Mandate / Description (JSON Object - focus on 'description', 'required_skills', 'qualifications', 'salary_text', 'benefits_text' fields if they exist, otherwise use the full object):
---
{job_data_json}
---

Based on the comparison, provide a suitability score from 0 to 100, where 100 is a perfect match.
Also provide a justification and fill in the other fields as accurately as possible.

Output *only* a valid JSON object matching the structure below. Do not include any introductory text, explanations, or markdown formatting.

JSON Structure to populate:
{{
  "suitability_score": "integer (0-100)",
  "justification": "string (Detailed explanation for the score, highlighting strengths and weaknesses regarding skills, experience, qualifications)",
  "skill_match": "boolean (Do the candidate's skills strongly align with the *required* skills mentioned in the job? Consider keywords and context.)",
  "experience_match": "boolean (Does the candidate's total years and type of experience align with the job requirements?)",
  "qualification_match": "boolean (Does the candidate meet the essential qualifications like degrees or certifications mentioned?)",
  "salary_alignment": "string (Assess potential salary fit based on experience level and any salary data provided in the job. Options: 'Likely Fit', 'Potentially Below Range', 'Potentially Above Range', 'Insufficient Data')",
  "benefit_alignment": "string (Briefly note any alignment or mention of key benefits if discussed in both resume context and job. E.g., 'Mentions Health/Retirement', 'Standard Benefits Likely', 'Insufficient Data')",
  "missing_keywords": ["string (List a few key *required* skills or qualifications from the job description that seem absent or weak in the resume)"]
}}

Analysis JSON:
"""