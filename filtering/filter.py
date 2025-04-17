import logging
from typing import Dict, Optional, List, Any
from .filter_utils import parse_salary, normalize_string

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def apply_filters(
    jobs: List[Dict[str, Any]],
    salary_min: Optional[int] = None,
    salary_max: Optional[int] = None,
    locations: Optional[List[str]] = None,
    work_models: Optional[List[str]] = None,
    job_types: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Filters a list of job dictionaries based on provided criteria.

    Args:
        jobs: The list of job dictionaries to filter.
        salary_min: Minimum desired annual salary.
        salary_max: Maximum desired annual salary.
        locations: List of desired locations (case-insensitive). "Remote" is handled specially.
        work_models: List of desired work models (e.g., ["Remote", "Hybrid"], case-insensitive).
        job_types: List of desired job types (e.g., ["Full-time"], case-insensitive).

    Returns:
        A new list containing only the jobs that match the filter criteria.
    """
    filtered_jobs = []
    normalized_locations = [normalize_string(loc) for loc in locations] if locations else []
    normalized_work_models = [normalize_string(wm) for wm in work_models] if work_models else []
    normalized_job_types = [normalize_string(jt) for jt in job_types] if job_types else []

    # import html
    logging.info("Applying filters: Min Salary=%s, Max Salary=%s, Locations=%s, Work Models=%s, Job Types=%s",
                 html.escape(str(salary_min)), html.escape(str(salary_max)),
                 html.escape(str(normalized_locations)), html.escape(str(normalized_work_models)),
                 html.escape(str(normalized_job_types)))

    for job in jobs:
        if not isinstance(job, dict):
            logging.warning("Skipping non-dictionary item in job list.")
            continue

        # --- Salary Filter ---
        # Try getting pre-parsed fields first, then fall back to parsing text
        job_min_salary = job.get('salary_min') or job.get('min_salary')
        job_max_salary = job.get('salary_max') or job.get('max_salary')
        salary_text = job.get('salary_text') or job.get('salary') # Common field names

        if job_min_salary is None and job_max_salary is None and isinstance(salary_text, str):
             # Parse from text only if structured fields are missing
             job_min_salary, job_max_salary = parse_salary(salary_text)
             logging.debug(f"Parsed salary for job '{job.get('title', '')}': Min={job_min_salary}, Max={job_max_salary}")  # import html


        salary_passes = True
        if salary_min is not None:
            # Job must have a max salary >= filter min, OR a min salary >= filter min
            if job_max_salary is not None and job_max_salary < salary_min:
                 salary_passes = False
            elif job_min_salary is not None and job_max_salary is None and job_min_salary < salary_min:
                 # Only min provided, and it's below filter min
                 salary_passes = False
            elif job_min_salary is None and job_max_salary is None:
                 # No salary info in job, cannot filter by min - pass for now? Or fail? Let's pass.
                 pass


        if salary_max is not None and salary_passes:
             # Job must have a min salary <= filter max, OR a max salary <= filter max
             if job_min_salary is not None and job_min_salary > salary_max:
                 salary_passes = False
             elif job_max_salary is not None and job_min_salary is None and job_max_salary > salary_max:
                 # Only max provided, and it's above filter max
                 salary_passes = False
             elif job_min_salary is None and job_max_salary is None:
                 # No salary info, pass filter
                 pass

        if not salary_passes:
            logging.debug(f"Job '{job.get('title', 'N/A')}' failed SALARY filter.")
            continue # Skip to next job

        # --- Location Filter ---
        location_passes = True
        if normalized_locations:
            job_location_text = normalize_string(job.get('location'))
            job_is_remote = 'remote' in job_location_text or normalize_string(job.get('work_model')) == 'remote'

            # If "remote" is a desired location, check if job is remote
            if 'remote' in normalized_locations and job_is_remote:
                location_passes = True # Matches remote requirement
            else:
                # Check if any part of the job location string matches any desired location
                location_found = False
                for loc in normalized_locations:
                    if loc != 'remote' and loc in job_location_text: # Simple substring check
                         location_found = True
                         break
                if not location_found and not ('remote' in normalized_locations and job_is_remote):
                     location_passes = False # No match including remote special case

        if not location_passes:
            logging.debug("Job '{}' failed LOCATION filter.".format(job.get('title', 'N/A')))  # Use string formatting to avoid potential log injection
            continue

        # --- Work Model Filter ---
        work_model_passes = True
        if normalized_work_models:
            job_model = normalize_string(job.get('work_model')) or normalize_string(job.get('remote')) # Common field names
            if not job_model: # Infer from location if possible
                 job_location_text = normalize_string(job.get('location'))
                 if 'remote' in job_location_text:
                     job_model = 'remote'
                 elif 'hybrid' in job_location_text:
                      job_model = 'hybrid'
                 elif 'on-site' in job_location_text or 'office' in job_location_text:
                      job_model = 'on-site'
                 # If still no model, assume it doesn't match specific filters? Pass for now.

            if job_model and job_model not in normalized_work_models:
                work_model_passes = False

        if not work_model_passes:
             logging.debug("Job '{}' failed WORK MODEL filter.".format(job.get('title', 'N/A')))  # Use string formatting to avoid potential log injection
             continue

        # --- Job Type Filter ---
        job_type_passes = True
        if normalized_job_types:
            job_type = normalize_string(job.get('job_type')) or normalize_string(job.get('employment_type'))
            # Add more potential field names if needed
            if not job_type:
                 # Try to infer from title/description? Difficult. Assume pass if not specified.
                 pass
            elif job_type and job_type not in normalized_job_types:
                job_type_passes = False

        if not job_type_passes:
            logging.debug("Job '{}' failed JOB TYPE filter.".format(job.get('title', 'N/A')))  # Use string formatting to avoid potential log injection
            continue

        # If all filters pass, add the job
        filtered_jobs.append(job)

    logging.info("Filtering complete. %d out of %d jobs passed filters.", len(filtered_jobs), len(jobs))  # Use % formatting to prevent log injection
    return filtered_jobs