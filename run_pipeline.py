# run_pipeline.py
import pandas as pd
import json
import logging
from datetime import datetime
import yaml # Ensure yaml is imported
import argparse # Ensure argparse is imported
import os
from typing import Dict, Any, Optional, List

# Import your other modules
from jobspy import scrape_jobs # Or your scraping function
from analysis.analyzer import JobAnalyzer # Assuming analyzer class
from main_matcher import load_and_extract_resume, analyze_jobs # Functions doing the core work
from utils.logging_setup import setup_logging # Your logging setup
from models import CombinedJobResult, OriginalJobData # Import Pydantic models

# Assuming logger is configured via setup_logging()

def load_config(config_path='config.yaml') -> Optional[Dict[str, Any]]:
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        # --- Add USER_PROFILE Loading ---
        user_profile = config.get('USER_PROFILE', {})
        logger.info(f"Loaded User Profile: Desired Salary ${user_profile.get('DESIRED_SALARY_MIN', 'N/A')}-${user_profile.get('DESIRED_SALARY_MAX', 'N/A')}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading configuration: {e}", exc_info=True)
        return None

def scrape_jobs_with_jobspy(
    search_term: str,
    location: str,
    sites: list[str],
    results_wanted: int,
    hours_old: int,
    country_indeed: str,
    offset: int = 0,
) -> Optional[pd.DataFrame]:
    """Scrapes jobs using JobSpy and handles potential errors."""
    logger.info("[bold blue]Starting job scraping via JobSpy...[/bold blue]")
    # ... (rest of your existing scrape_jobs_with_jobspy function) ...
    # Ensure it returns a DataFrame or None
    try:
        jobs_df = scrape_jobs(
            site_name=sites,
            search_term=search_term,
            location=location,
            results_wanted=results_wanted,
            hours_old=hours_old,
            country_indeed=country_indeed,
            offset=offset,
            linkedin_company_ids=None,
            linkedin_fetch_description=True# Ensure this is None or omitted
        )
        if jobs_df is not None and not jobs_df.empty:
            logger.info(f"Jobspy scraping successful. Found {len(jobs_df)} jobs.")
            return jobs_df
        else:
            logger.warning("Jobspy scraping returned no results or an empty DataFrame.")
            return None
    except Exception as e:
        logger.error(f"An error occurred during jobspy scraping: {e}", exc_info=True)
        return None


def process_scraped_data(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Converts DataFrame, adds metadata, and normalizes fields."""
    if df is None or df.empty:
        return []

    processed_list = []
    df_copy = df.copy() # Work on a copy

    # Rename columns if needed (adjust based on your actual needs)
    rename_map = {'job_url': 'url', 'job_type': 'employment_type'}
    df_copy.rename(columns=rename_map, inplace=True)
    logger.debug(f"Renamed DataFrame columns: {rename_map}")

    # Add missing columns expected by OriginalJobData model if they don't exist
    expected_cols = OriginalJobData.model_fields.keys()
    for col in expected_cols:
         if col not in df_copy.columns and col not in ['salary_estimated', 'normalized_date_posted']:
              df_copy[col] = None # Add missing columns with None
              logger.warning(f"Column '{col}' missing from scraped data, adding as empty.")


    # Fill NA/NaN values - Use empty string or specific defaults
    # Careful with types - fillna might change dtype if not careful
    for col in df_copy.columns:
        if df_copy[col].isnull().any():
             if pd.api.types.is_numeric_dtype(df_copy[col]):
                  # Decide numeric fill value (e.g., 0, None, or leave as NaN temporarily)
                  pass # Pydantic handles Optional[float] = None
             elif pd.api.types.is_bool_dtype(df_copy[col]):
                  pass # Pydantic handles Optional[bool] = None
             else:
                  df_copy[col] = df_copy[col].fillna('') # Fill string/object NaNs
    logger.debug("Attempted to fill NA/NaN values.")


    # Process rows
    for _, row in df_copy.iterrows():
        job_dict = row.to_dict()

        # --- Add Salary Estimated Flag ---
        job_dict['salary_estimated'] = job_dict.get('salary_source', '') not in ['direct_data', 'employer_estimate', '']

        # --- Normalize Date Posted ---
        date_str = job_dict.get('date_posted')
        normalized_date = None
        if date_str and isinstance(date_str, str):
            try:
                # Attempt to parse common date formats (add more as needed)
                 normalized_date = pd.to_datetime(date_str).to_pydatetime()
            except Exception:
                 logger.warning(f"Could not parse date_posted: {date_str}", exc_info=False)
                 # Keep original string if parsing fails
                 job_dict['date_posted'] = date_str # Ensure original is kept
                 normalized_date = None
        elif isinstance(date_str, datetime): # If JobSpy already parsed it
             normalized_date = date_str

        job_dict['normalized_date_posted'] = normalized_date

        # Convert types for Pydantic validation where needed (e.g., if numbers came as strings)
        # Example:
        for key in ['min_amount', 'max_amount', 'company_rating', 'company_reviews_count', 'vacancy_count']:
             if key in job_dict and job_dict[key] == '':
                   job_dict[key] = None # Explicitly set empty strings for numeric/int Optional fields to None
             elif key in job_dict and job_dict[key] is not None:
                  try:
                       if key in ['min_amount', 'max_amount', 'company_rating']:
                            job_dict[key] = float(job_dict[key])
                       elif key in ['company_reviews_count', 'vacancy_count']:
                            job_dict[key] = int(float(job_dict[key])) # Handle potential floats like '5.0'
                  except (ValueError, TypeError):
                       logger.warning(f"Could not convert {key} value '{job_dict[key]}' to numeric. Setting to None.")
                       job_dict[key] = None

        processed_list.append(job_dict)

    logger.debug(f"Converted DataFrame to list of {len(processed_list)} dictionaries.")
    return processed_list

def run_pipeline(args, config):
    """Main pipeline execution function."""
    logger.info(f"[bold green]Starting Pipeline Run[/bold green] ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")

    # --- Get User Profile from Config ---
    user_profile = config.get('USER_PROFILE', {})
    logger.info(f"Using User Profile - Desired Salary: ${user_profile.get('DESIRED_SALARY_MIN', 'N/A')}-${user_profile.get('DESIRED_SALARY_MAX', 'N/A')}, Must-Haves: {user_profile.get('MUST_HAVE_SKILLS', [])}")


    # --- 1. Scraping ---
    scraped_jobs_df = scrape_jobs_with_jobspy(
        search_term=args.search,
        location=args.location,
        sites=args.sites.split(',') if args.sites else config.get('SCRAPING', {}).get('SITES', []),
        results_wanted=args.results if args.results else config.get('SCRAPING', {}).get('RESULTS_WANTED', 50),
        hours_old=args.hours_old if args.hours_old else config.get('SCRAPING', {}).get('HOURS_OLD', 72),
        country_indeed=args.country_indeed if args.country_indeed else config.get('SCRAPING', {}).get('COUNTRY_INDEED', 'US')
    )

    if scraped_jobs_df is None or scraped_jobs_df.empty:
        logger.warning("Scraping yielded no results. Pipeline cannot continue.")
        # Optionally create an empty results file or just exit
        save_analysis_results([], args.analysis_output) # Save empty list
        return

    # --- 2. Deduplication (A4) ---
    initial_count = len(scraped_jobs_df)
    # Prioritize direct URL if available, otherwise use combination
    subset_cols = ['title', 'company', 'location'] # Base subset
    if 'job_url_direct' in scraped_jobs_df.columns and scraped_jobs_df['job_url_direct'].notna().any():
         # Fill NA in job_url_direct temporarily for hashing/comparison if needed, or use it as primary key
         # A simpler approach: Use URL when available, otherwise fall back to combo
         # This requires more complex logic, sticking to simple subset for now:
         if scraped_jobs_df['job_url_direct'].nunique() > 0.5 * initial_count: # Heuristic: if many unique URLs exist
              subset_cols = ['job_url_direct'] + subset_cols # Prioritize URL but include others as fallback
              df_unique = scraped_jobs_df.drop_duplicates(subset=['job_url_direct'], keep='first')
              logger.info(f"Attempting deduplication primarily based on 'job_url_direct'.")
         else:
              logger.info(f"Attempting deduplication based on 'title', 'company', 'location'.")
              df_unique = scraped_jobs_df.drop_duplicates(subset=subset_cols, keep='first')

    else: # job_url_direct doesn't exist or is all null
         logger.info(f"Attempting deduplication based on 'title', 'company', 'location'.")
         df_unique = scraped_jobs_df.drop_duplicates(subset=subset_cols, keep='first')

    duplicates_removed = initial_count - len(df_unique)
    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed} duplicate job listings.")
    else:
        logger.info("No duplicate job listings found based on selected criteria.")

    # --- 3. Process Scraped Data (A2) ---
    jobs_list_dict = process_scraped_data(df_unique)

    # --- 4. Save Raw/Processed Scraped Data (Optional but Recommended) ---
    raw_output_path = config.get('OUTPUT', {}).get('SCRAPED_JOBS_JSON', './output/scraped_jobs.json')
    try:
        os.makedirs(os.path.dirname(raw_output_path), exist_ok=True)
        with open(raw_output_path, 'w', encoding='utf-8') as f:
            json.dump(jobs_list_dict, f, indent=4, default=str) # Use default=str for datetime etc.
        logger.info(f"Saved {len(jobs_list_dict)} processed scraped jobs to {raw_output_path}")
    except Exception as e:
        logger.error(f"Failed to save scraped jobs JSON: {e}", exc_info=True)


    # --- 5. Load Resume & Initialize Analyzer ---
    analyzer = JobAnalyzer(
        model_name=args.ollama_model if args.ollama_model else config.get('AI_PROCESSING', {}).get('OLLAMA_MODEL'),
        ollama_base_url=config.get('AI_PROCESSING', {}).get('OLLAMA_BASE_URL'),
        resume_prompt_path=config.get('AI_PROCESSING', {}).get('PROMPTS', {}).get('RESUME_EXTRACTION'),
        suitability_prompt_path=config.get('AI_PROCESSING', {}).get('PROMPTS', {}).get('JOB_SUITABILITY')
    )

    if not analyzer.check_connection():
        logger.error("Ollama connection failed. Cannot proceed with analysis.")
        # Decide behavior: maybe save just scraped data or exit
        save_analysis_results([], args.analysis_output) # Save empty list
        return

    resume_filepath = args.resume if args.resume else config.get('AI_PROCESSING', {}).get('RESUME_FILEPATH')
    structured_resume = load_and_extract_resume(resume_filepath, analyzer)

    if not structured_resume:
        logger.error("Failed to extract structured data from resume. Cannot proceed with analysis.")
        save_analysis_results([], args.analysis_output) # Save empty list
        return

    # --- 6. Analyze Jobs ---
    analyzed_results: List[CombinedJobResult] = analyze_jobs(analyzer, structured_resume, jobs_list_dict, user_profile)

    # --- 7. Save Analysis Results ---
    save_analysis_results(analyzed_results, args.analysis_output)

    logger.info(f"[bold green]Pipeline Run Finished[/bold green] ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")


def save_analysis_results(results: List[CombinedJobResult], filename: str):
    """Saves the final combined job data and analysis results to JSON."""
    logger.info(f"Saving final analysis results to {filename}...")
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # Convert Pydantic models to dicts for JSON serialization
        output_data = [result.model_dump(mode='json') for result in results]

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4) # Pydantic handles datetime serialization in mode='json'
        logger.info(f"Saved {len(results)} analyzed jobs to {filename}")
    except Exception as e:
        logger.error(f"Failed to save analysis results JSON: {e}", exc_info=True)
        # Optionally save an empty file if saving fails mid-way or results are empty
        if not results:
             try:
                  with open(filename, 'w', encoding='utf-8') as f:
                       json.dump([], f, indent=4)
                  logger.info(f"Empty analysis results file created at {filename}")
             except Exception as empty_save_err:
                  logger.error(f"Failed even to save empty results file: {empty_save_err}")


def main():
    # --- Setup Logging ---
    setup_logging() # Call your logging setup function

    # --- Load Base Config ---
    config = load_config()
    if not config:
        logger.critical("Exiting due to config loading failure.")
        return

    # --- Setup Argument Parser ---
    parser = argparse.ArgumentParser(description="MyJobSpy - Scrape and analyze job postings.")
    # Add arguments (search, location, resume, sites, results, hours_old, country_indeed, ollama_model, analysis_output, verbose -v)
    parser.add_argument("--search", type=str, help="Job search keywords (overrides config)")
    parser.add_argument("--location", type=str, help="Job search location (overrides config)")
    parser.add_argument("--resume", type=str, help="Path to resume PDF (overrides config)")
    parser.add_argument("--sites", type=str, help="Comma-separated list of sites (linkedin,indeed,etc) (overrides config)")
    parser.add_argument("--results", type=int, help="Approximate number of results per site (overrides config)")
    parser.add_argument("--hours-old", type=int, help="Max job age in hours (overrides config)")
    parser.add_argument("--country-indeed", type=str, help="Country for Indeed search (e.g., US, CA) (overrides config)")
    parser.add_argument("--ollama-model", type=str, help="Ollama model to use (overrides config)")
    parser.add_argument("--analysis-output", type=str, help="Output filename for analysis results (overrides config)")
    parser.add_argument("-v", "--verbose", action='store_true', help="Enable DEBUG logging level")

    args = parser.parse_args()

    # --- Update Log Level if Verbose ---
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled.")

    # --- Override Config with Args ---
    # Example: search_term = args.search or config.get('SEARCH_SETTINGS', {}).get('KEYWORDS', 'default')
    # Do this for all relevant args...
    final_args = argparse.Namespace(
         search=args.search or config.get('SEARCH_SETTINGS', {}).get('KEYWORDS', 'Software Engineer'),
         location=args.location or config.get('SEARCH_SETTINGS', {}).get('LOCATION', 'Remote'),
         resume=args.resume or config.get('AI_PROCESSING', {}).get('RESUME_FILEPATH'),
         sites=args.sites or ",".join(config.get('SCRAPING', {}).get('SITES', ["linkedin", "indeed"])),
         results=args.results or config.get('SCRAPING', {}).get('RESULTS_WANTED', 50),
         hours_old=args.hours_old or config.get('SCRAPING', {}).get('HOURS_OLD', 72),
         country_indeed=args.country_indeed or config.get('SCRAPING', {}).get('COUNTRY_INDEED', 'US'),
         ollama_model=args.ollama_model or config.get('AI_PROCESSING', {}).get('OLLAMA_MODEL'),
         analysis_output=args.analysis_output or config.get('OUTPUT', {}).get('ANALYSIS_RESULTS_JSON', './output/analysis_results.json'),
         verbose=args.verbose
    )

    # --- Validate Essential Args ---
    if not final_args.resume or not os.path.exists(final_args.resume):
         logger.critical(f"Resume file not found or not specified: {final_args.resume}. Exiting.")
         return
    if not final_args.ollama_model:
         logger.critical("Ollama model not specified in config or args. Exiting.")
         return

    # --- Run Pipeline ---
    try:
        run_pipeline(final_args, config)
    except Exception as e:
        logger.critical(f"An unexpected critical error occurred during pipeline execution: {e}", exc_info=True)
        # Optionally save partial results or state if possible

if __name__ == "__main__":
    main()