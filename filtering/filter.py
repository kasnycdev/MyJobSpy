import logging
from typing import Dict, Optional, List, Any
from .filter_utils import parse_salary, normalize_string

# Use root logger configured by run_pipeline or main_matcher
log = logging.getLogger(__name__)

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
    Expects job dictionaries based on scraped/converted data.
    """
    filtered_jobs = []
    normalized_locations = [normalize_string(loc) for loc in locations] if locations else []
    normalized_work_models = [normalize_string(wm) for wm in work_models] if work_models else []
    normalized_job_types = [normalize_string(jt) for jt in job_types] if job_types else []

    if not any([salary_min, salary_max, locations, work_models, job_types]):
         log.info("No filters provided, returning all jobs.")
         return jobs # Return early if no filters

    filter_summary = []
    if salary_min is not None: filter_summary.append(f"Min Salary >= {salary_min}")
    if salary_max is not None: filter_summary.append(f"Max Salary <= {salary_max}")
    if locations: filter_summary.append(f"Locations = {locations}")
    if work_models: filter_summary.append(f"Work Models = {work_models}")
    if job_types: filter_summary.append(f"Job Types = {job_types}")
    log.info(f"Applying filters: {', '.join(filter_summary)}")

    initial_count = len(jobs)
    for job in jobs:
        job_title = job.get('title', 'N/A') # For logging
        job_url = job.get('url', '#') # For logging

        if not isinstance(job, dict):
            log.warning("Skipping non-dictionary item in job list during filtering.")
            continue

        # --- Salary Filter ---
        job_min_salary, job_max_salary = None, None
        salary_text = job.get('salary_text') # Use the field name after conversion
        if isinstance(salary_text, str) and salary_text:
             job_min_salary, job_max_salary = parse_salary(salary_text)
             log.debug(f"Parsed salary for '{job_title}': Min={job_min_salary}, Max={job_max_salary} from '{salary_text}'")
        elif log.isEnabledFor(logging.DEBUG):
             # Log only in debug if salary text is missing/empty
             log.debug(f"No salary text found for '{job_title}'")


        salary_passes = True
        # Logic: If a filter is set, the job must have *some* data to compare against,
        # or it must explicitly meet the criteria if data exists.
        # Jobs without salary info will PASS salary filters by default.

        # Check MINIMUM salary requirement
        if salary_min is not None:
            # If job has salary info, check if it meets the minimum
            if job_max_salary is not None: # If job has max salary, it must be >= filter min
                if job_max_salary < salary_min: salary_passes = False
            elif job_min_salary is not None: # If job only has min salary, it must be >= filter min
                if job_min_salary < salary_min: salary_passes = False
            # If job has NO salary info (both None), it passes the MIN filter

        # Check MAXIMUM salary requirement (only if it passed MIN check)
        if salary_passes and salary_max is not None:
             # If job has salary info, check if it meets the maximum
             if job_min_salary is not None: # If job has min salary, it must be <= filter max
                 if job_min_salary > salary_max: salary_passes = False
             elif job_max_salary is not None: # If job only has max salary, it must be <= filter max
                 if job_max_salary > salary_max: salary_passes = False
             # If job has NO salary info (both None), it passes the MAX filter

        if not salary_passes:
            log.debug(f"FILTERED (Salary): '{job_title}' ({job_url})")
            continue

        # --- Location Filter ---
        location_passes = True
        if normalized_locations:
            job_location_text = normalize_string(job.get('location'))
            if not job_location_text:
                 location_passes = False # Job must have a location specified if filtering by location
            else:
                 # Check for remote explicitly first
                 job_is_remote = 'remote' in job_location_text
                 filter_wants_remote = 'remote' in normalized_locations

                 match_found = False
                 if filter_wants_remote and job_is_remote:
                      match_found = True
                 else:
                      # Check non-remote locations
                      for loc_filter in normalized_locations:
                           if loc_filter != 'remote' and loc_filter in job_location_text:
                                match_found = True
                                break
                 location_passes = match_found

        if not location_passes:
            log.debug(f"FILTERED (Location): '{job_title}' ({job_url})")
            continue

        # --- Work Model Filter ---
        work_model_passes = True
        if normalized_work_models:
            # Infer work model if not explicitly provided (basic inference)
            job_model_text = normalize_string(job.get('work_model'))
            if not job_model_text:
                job_loc = normalize_string(job.get('location'))
                if 'remote' in job_loc: job_model_text = 'remote'
                elif 'hybrid' in job_loc: job_model_text = 'hybrid'
                elif 'on-site' in job_loc or 'office' in job_loc: job_model_text = 'on-site'
            # Now check filter
            if not job_model_text:
                 work_model_passes = False # Job must have a model if filtering by model
            elif job_model_text not in normalized_work_models:
                work_model_passes = False

        if not work_model_passes:
             log.debug(f"FILTERED (Work Model): '{job_title}' ({job_url})")
             continue

        # --- Job Type Filter ---
        job_type_passes = True
        if normalized_job_types:
            job_type_text = normalize_string(job.get('employment_type')) # Use consistent field name
            if not job_type_text:
                 job_type_passes = False # Job must have a type if filtering by type
            elif job_type_text not in normalized_job_types:
                # Allow partial matches for flexibility (e.g., "Full-time" matches "Full-time Employee")
                partial_match = any(jt in job_type_text for jt in normalized_job_types)
                if not partial_match:
                     job_type_passes = False

        if not job_type_passes:
            log.debug(f"FILTERED (Job Type): '{job_title}' ({job_url})")
            continue

        # If all filters pass, add the job
        filtered_jobs.append(job)

    final_count = len(filtered_jobs)
    log.info(f"Filtering complete. {final_count} out of {initial_count} jobs passed filters.")
    return filtered_jobs