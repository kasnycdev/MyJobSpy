# run_pipeline.py

import argparse
import logging
import json
import os
import sys
from datetime import datetime
import pandas as pd

# Use the jobspy library for scraping
try:
    import jobspy
except ImportError:
    print("Error: 'jobspy' library not found. Please install it with 'pip install jobspy'")
    sys.exit(1)

# Import our analysis components (assuming main_matcher.py is in the same directory or PYTHONPATH)
# If main_matcher is not directly runnable as main(), we might need to refactor
# it slightly to expose an analysis function. For now, assume we call its main().
try:
    from main_matcher import main as run_analysis
except ImportError:
    print("Error: Could not import 'main' from 'main_matcher'. Ensure it's in the correct path.")
    sys.exit(1)


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING) # Reduce httpx verbosity often seen with playwright/jobspy


def scrape_jobs_with_jobspy(
    search_terms: str,
    location: str,
    sites: list[str],
    results_wanted: int,
    hours_old: int,
    country_indeed: str,
    proxies: Optional[list[str]] = None, # Added proxy support
    offset: int = 0
    ) -> Optional[pd.DataFrame]:
    """
    Uses the jobspy library to scrape jobs.

    Returns:
        A pandas DataFrame with scraped jobs, or None on failure.
    """
    logging.info(f"Starting jobspy scraping for '{search_terms}' in '{location}' on sites: {sites}")
    logging.info(f"Parameters: Results wanted={results_wanted}, Hours old={hours_old}, Country={country_indeed}, Offset={offset}")

    try:
        jobs_df = jobspy.scrape_jobs(
            site_name=sites,
            search_term=search_terms,
            location=location,
            results_wanted=results_wanted,
            hours_old=hours_old,
            country_indeed=country_indeed, # Use 'usa' for US indeed, etc.
            proxies=proxies,
            offset=offset
            # Add other jobspy parameters as needed (e.g., verbose=True, description_format='markdown')
            # description_format='html' might be better for LLM parsing if 'markdown' is too lossy
        )

        if jobs_df is None or jobs_df.empty:
            logging.warning("Jobspy scraping returned no results or failed silently.")
            return None
        else:
            logging.info(f"Jobspy scraping successful. Found {len(jobs_df)} potential jobs.")
            # Basic check for essential columns (adjust if needed based on jobspy output)
            required_cols = ['title', 'company', 'location', 'description', 'job_url']
            if not all(col in jobs_df.columns for col in required_cols):
                 logging.warning(f"Jobspy output DataFrame might be missing expected columns. Found: {jobs_df.columns.tolist()}")
            return jobs_df

    except Exception as e:
        logging.error(f"An error occurred during jobspy scraping: {e}", exc_info=True) # Log traceback
        # Specific errors might need more nuanced handling (e.g., Playwright timeouts)
        return None


def main():
    parser = argparse.ArgumentParser(description="Run Job Scraping (via JobSpy) and GenAI Analysis Pipeline.")

    # --- Scraping Arguments (Leveraging JobSpy options) ---
    parser.add_argument("--search", required=True, help="Job title, keywords, or company to search for.")
    parser.add_argument("--location", default="Remote", help="Location to search for jobs (e.g., 'New York', 'Remote').")
    # Updated site list based on common jobspy support
    parser.add_argument("--sites", default="linkedin,indeed,zip_recruiter,glassdoor",
                        help="Comma-separated list of sites supported by jobspy (e.g., linkedin,indeed,zip_recruiter,glassdoor,dice,google).")
    parser.add_argument("--results", type=int, default=20, help="Approximate total number of jobs to scrape across all sites.")
    parser.add_argument("--hours-old", type=int, default=72, help="Filter jobs posted within the last N hours (e.g., 24 for past day). Set 0 to disable.")
    parser.add_argument("--country-indeed", default="usa", help="Country for Indeed search (e.g., 'usa', 'uk', 'ca'). See jobspy docs.")
    parser.add_argument("--proxies", help="Comma-separated list of proxies to use (e.g., 'http://user:pass@host:port,...').")
    parser.add_argument("--offset", type=int, default=0, help="Offset for search results (e.g., start from the 10th result).")
    parser.add_argument("--scraped-jobs-file", default="output/scraped_jobs.json", help="Intermediate file to save scraped job data in JSON format.")


    # --- Analysis Arguments (Passed to main_matcher) ---
    parser.add_argument("--resume", required=True, help="Path to the resume file (.docx or .pdf)")
    parser.add_argument("--analysis-output", default="output/analyzed_jobs.json", help="Path for the final analysis output JSON file.")
    # Filtering args remain the same as they are applied *after* scraping during analysis
    parser.add_argument("--min-salary", type=int, help="Minimum desired annual salary for filtering.")
    parser.add_argument("--max-salary", type=int, help="Maximum desired annual salary for filtering.")
    parser.add_argument("--filter-locations", help="Comma-separated list of desired locations for filtering (e.g., 'New York,Remote')")
    parser.add_argument("--filter-work-models", help="Comma-separated list of desired work models for filtering (e.g., 'Remote,Hybrid')")
    parser.add_argument("--filter-job-types", help="Comma-separated list of desired job types for filtering (e.g., 'Full-time')")

    args = parser.parse_args()

    # --- Step 1: Run Scraper using JobSpy ---
    scraper_sites = [site.strip() for site in args.sites.split(',')]
    output_dir_scraped = os.path.dirname(args.scraped_jobs_file)
    if output_dir_scraped:
            os.makedirs(output_dir_scraped, exist_ok=True)

    proxy_list = [p.strip() for p in args.proxies.split(',')] if args.proxies else None

    jobs_df = scrape_jobs_with_jobspy(
        search_terms=args.search,
        location=args.location,
        sites=scraper_sites,
        results_wanted=args.results,
        hours_old=args.hours_old,
        country_indeed=args.country_indeed,
        proxies=proxy_list,
        offset=args.offset
    )

    if jobs_df is None or jobs_df.empty:
        logging.warning(f"Scraping yielded no results. No analysis will be performed.")
        # Create an empty analysis output file
        output_dir_analysis = os.path.dirname(args.analysis_output)
        if output_dir_analysis:
            os.makedirs(output_dir_analysis, exist_ok=True)
        with open(args.analysis_output, 'w', encoding='utf-8') as f:
            json.dump([], f)
        logging.info(f"Empty analysis results saved to {args.analysis_output}")
        return

    # --- Step 2: Convert DataFrame to List of Dictionaries and Save ---
    # Ensure consistency in essential fields for the analyzer
    # Jobspy might have slightly different column names, adjust as needed
    # Example mapping (check actual jobspy output columns):
    jobs_df = jobs_df.rename(columns={'job_url': 'url', 'job_type': 'employment_type'}) # Adjust based on actual columns

    # Fill missing essential columns with None or empty strings if necessary
    for col in ['description', 'location', 'salary', 'employment_type']: # Add other fields checked by filter/analyzer
         if col not in jobs_df.columns:
              jobs_df[col] = None

    # Handle potential NaN values which are not JSON serializable
    jobs_df = jobs_df.fillna('') # Replace NaN with empty string, or use None if preferred

    jobs_list = jobs_df.to_dict('records')

    try:
        with open(args.scraped_jobs_file, 'w', encoding='utf-8') as f:
            json.dump(jobs_list, f, indent=4)
        logging.info(f"Saved {len(jobs_list)} scraped jobs to {args.scraped_jobs_file}")
    except Exception as e:
        logging.error(f"Error saving scraped jobs to JSON file {args.scraped_jobs_file}: {e}")
        return # Stop if we cannot save the intermediate file

    # --- Step 3: Run Analysis ---
    logging.info(f"Starting analysis using resume '{args.resume}' and jobs file '{args.scraped_jobs_file}'...")

    # Prepare arguments for the analysis script (main_matcher.main)
    analysis_args = [
        '--resume', args.resume,
        '--jobs', args.scraped_jobs_file, # Use the saved JSON file from jobspy results
        '--output', args.analysis_output
    ]
    # Append filtering arguments
    if args.min_salary is not None: analysis_args.extend(['--min-salary', str(args.min_salary)])
    if args.max_salary is not None: analysis_args.extend(['--max-salary', str(args.max_salary)])
    if args.filter_locations: analysis_args.extend(['--locations', args.filter_locations])
    if args.filter_work_models: analysis_args.extend(['--work-models', args.filter_work_models])
    if args.filter_job_types: analysis_args.extend(['--job-types', args.filter_job_types])

    # Temporarily modify sys.argv and call the main_matcher's main function
    original_argv = sys.argv
    sys.argv = [sys.argv[0]] + analysis_args # Simulate command line args for main_matcher
    try:
        logging.info(f"Calling analysis script with args: {analysis_args}")
        run_analysis() # Call the imported main function
        logging.info("Analysis process completed.")
    except SystemExit as e:
         logging.error(f"Analysis script exited with code: {e.code}")
    except Exception as e:
        logging.error(f"An error occurred during the analysis step: {e}", exc_info=True)
    finally:
        sys.argv = original_argv # Restore original sys.argv

if __name__ == "__main__":
    main()