import argparse
import logging
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

# Use the jobspy library for scraping
try:
    import jobspy
    from jobspy import scrape_jobs # Explicit import
except ImportError:
    print("CRITICAL ERROR: 'jobspy' library not found. Please install it via requirements.txt")
    sys.exit(1)

# Import analysis components from main_matcher
try:
    from main_matcher import load_and_extract_resume, analyze_jobs, apply_filters_sort_and_save
    from analysis.analyzer import ResumeAnalyzer # Need analyzer class
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import analysis functions from main_matcher or analyzer: {e}")
    print("Ensure main_matcher.py and analysis/analyzer.py are in the correct path.")
    sys.exit(1)

import config # Central configuration

# Rich for UX
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

# Setup logging using Rich
logging.basicConfig(
    level=config.LOG_LEVEL,
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATE_FORMAT,
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)] # show_path=False for cleaner logs
)
logging.getLogger("httpx").setLevel(logging.WARNING) # Reduce httpx verbosity
log = logging.getLogger(__name__) # Use project logger
console = Console()


def scrape_jobs_with_jobspy(
    search_terms: str,
    location: str,
    sites: list[str],
    results_wanted: int,
    hours_old: int,
    country_indeed: str,
    proxies: Optional[list[str]] = None,
    offset: int = 0
    ) -> Optional[pd.DataFrame]:
    """Uses the jobspy library to scrape jobs, with better logging."""
    log.info(f"[bold blue]Starting job scraping via JobSpy...[/bold blue]")
    log.info(f"Search: '[cyan]{search_terms}[/cyan]' | Location: '[cyan]{location}[/cyan]' | Sites: {sites}")
    log.info(f"Params: Results ≈{results_wanted}, Max Age={hours_old}h, Indeed Country='{country_indeed}', Offset={offset}")
    if proxies:
        log.info(f"Using {len(proxies)} proxies.")

    try:
        # Note: verbose=1 for jobspy might show progress within its own logs
        jobs_df = scrape_jobs(
            site_name=sites,
            search_term=search_terms,
            location=location,
            results_wanted=results_wanted,
            hours_old=hours_old,
            country_indeed=country_indeed,
            proxies=proxies,
            offset=offset,
            verbose=1, # Show jobspy's internal logging/progress
            description_format="markdown" # Or 'html' - markdown is usually cleaner
        )

        if jobs_df is None or jobs_df.empty:
            log.warning("Jobspy scraping returned no results or failed.")
            return None
        else:
            log.info(f"Jobspy scraping successful. Found {len(jobs_df)} jobs.")
            log.debug(f"DataFrame columns: {jobs_df.columns.tolist()}")
            # Check essential columns
            required_cols = ['title', 'company', 'location', 'description', 'job_url']
            missing_cols = [col for col in required_cols if col not in jobs_df.columns]
            if missing_cols:
                 log.warning(f"Jobspy output DataFrame is missing expected columns: {missing_cols}")
            return jobs_df

    except ImportError as ie:
         log.critical(f"Import error during scraping: {ie}. Ensure all jobspy dependencies are installed.")
         return None
    except Exception as e:
        log.error(f"An error occurred during jobspy scraping: {e}", exc_info=True)
        return None


def convert_and_save_scraped(jobs_df: pd.DataFrame, output_path: str) -> List[Dict[str, Any]]:
    """Converts DataFrame to list of dicts and saves to JSON."""
    log.info(f"Converting DataFrame to list and saving to {output_path}")

    # Rename columns for consistency with analysis expectations (adjust if jobspy changes)
    rename_map = {
        'job_url': 'url',
        'job_type': 'employment_type', # Example, check jobspy output
        'salary': 'salary_text',       # Pass raw salary text for parsing/LLM
        'benefits': 'benefits_text'    # Pass raw benefits text
    }
    # Only rename columns that actually exist in the DataFrame
    actual_rename_map = {k: v for k, v in rename_map.items() if k in jobs_df.columns}
    if actual_rename_map:
        jobs_df = jobs_df.rename(columns=actual_rename_map)
        log.debug(f"Renamed DataFrame columns: {actual_rename_map}")

    # Ensure essential columns for filtering/analysis exist, fill with '' if missing
    essential_cols = ['title', 'company', 'location', 'description', 'url',
                      'salary_text', 'employment_type', 'benefits_text', 'skills'] # Added 'skills'
    for col in essential_cols:
         if col not in jobs_df.columns:
              log.warning(f"Column '{col}' missing from scraped data, adding as empty.")
              jobs_df[col] = '' # Use empty string instead of None for easier JSON handling

    # Convert NaN/NaT to empty string for JSON compatibility
    jobs_df = jobs_df.fillna('')

    # Convert DataFrame to list of dictionaries
    jobs_list = jobs_df.to_dict('records')

    # Save to JSON
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(jobs_list, f, indent=4)
        log.info(f"Saved {len(jobs_list)} scraped jobs to {output_path}")
        return jobs_list
    except Exception as e:
        log.error(f"Error saving scraped jobs to JSON file {output_path}: {e}", exc_info=True)
        return [] # Return empty list on save failure


def print_summary_table(results_json: List[Dict[str, Any]], top_n: int = 10):
    """Prints a summary table of top results using Rich."""
    if not results_json:
        console.print("[yellow]No analysis results to summarize.[/yellow]")
        return

    table = Table(title=f"Top {min(top_n, len(results_json))} Job Matches", show_header=True, header_style="bold magenta", show_lines=False)
    table.add_column("Score", style="dim", width=6, justify="right")
    table.add_column("Title", style="bold", min_width=20)
    table.add_column("Company")
    table.add_column("Location")
    table.add_column("URL", overflow="fold", style="cyan") # Fold long URLs

    # Results should already be sorted, but access nested data safely
    count = 0
    for result in results_json:
        if count >= top_n:
            break

        analysis = result.get('analysis', {})
        original = result.get('original_job_data', {})

        score = analysis.get('suitability_score', -1)
        if score == -1: continue # Skip failed analysis entries in summary

        score_str = f"{score}%"
        table.add_row(
            score_str,
            original.get('title', 'N/A'),
            original.get('company', 'N/A'),
            original.get('location', 'N/A'),
            original.get('url', '#')
        )
        count += 1

    if count == 0:
         console.print("[yellow]No successfully analyzed jobs to display in summary.[/yellow]")
    else:
         console.print(table)


# --- Main Execution ---
def run_pipeline():
    parser = argparse.ArgumentParser(
        description="Run Job Scraping (via JobSpy) and GenAI Analysis Pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show default values
    )

    # --- Scraping Arguments ---
    scrape_group = parser.add_argument_group('Scraping Options (JobSpy)')
    scrape_group.add_argument("--search", required=True, help="Job title, keywords, or company to search for.")
    scrape_group.add_argument("--location", default="Remote", help="Location to search (e.g., 'New York', 'Remote').")
    scrape_group.add_argument("--sites", default=",".join(config.DEFAULT_SCRAPE_SITES),
                              help="Comma-separated sites (e.g., linkedin,indeed,zip_recruiter). Check JobSpy docs.")
    scrape_group.add_argument("--results", type=int, default=config.DEFAULT_RESULTS_LIMIT, help="Approx total jobs to fetch.")
    scrape_group.add_argument("--hours-old", type=int, default=config.DEFAULT_HOURS_OLD, help="Max job age in hours (0=disable).")
    scrape_group.add_argument("--country-indeed", default=config.DEFAULT_COUNTRY_INDEED, help="Country for Indeed search ('usa', 'uk', etc.).")
    scrape_group.add_argument("--proxies", help="Comma-separated proxies ('http://user:pass@host:port,...').")
    scrape_group.add_argument("--offset", type=int, default=0, help="Search results offset.")
    scrape_group.add_argument("--scraped-jobs-file", default=config.DEFAULT_SCRAPED_JSON, help="Intermediate file for scraped jobs.")

    # --- Analysis Arguments ---
    analysis_group = parser.add_argument_group('Analysis Options')
    analysis_group.add_argument("--resume", required=True, help="Path to the resume file (.docx or .pdf)")
    analysis_group.add_argument("--analysis-output", default=config.DEFAULT_ANALYSIS_JSON, help="Final analysis output JSON file.")
    analysis_group.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG level logging.")

    # --- Filtering Arguments ---
    filter_group = parser.add_argument_group('Filtering Options (Applied After Analysis)')
    filter_group.add_argument("--min-salary", type=int, help="Minimum desired annual salary.")
    filter_group.add_argument("--max-salary", type=int, help="Maximum desired annual salary.")
    filter_group.add_argument("--filter-locations", help="Comma-separated locations (e.g., 'New York,Remote')")
    filter_group.add_argument("--filter-work-models", help="Comma-separated work models (e.g., 'Remote,Hybrid')")
    filter_group.add_argument("--filter-job-types", help="Comma-separated job types (e.g., 'Full-time')")


    args = parser.parse_args()

    # --- Setup Logging Level ---
    log_level = logging.DEBUG if args.verbose else config.LOG_LEVEL
    logging.getLogger().setLevel(log_level) # Set root logger level

    log.info(f"[bold green]Starting Pipeline Run[/bold green] ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    config.ensure_output_dir() # Ensure output dir exists

    # --- Step 1: Scrape Jobs ---
    scraper_sites = [site.strip().lower() for site in args.sites.split(',')]
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
        log.warning("Scraping yielded no results. Pipeline cannot continue.")
        # Create empty analysis output file for consistency
        with open(args.analysis_output, 'w', encoding='utf-8') as f: json.dump([], f)
        log.info(f"Empty analysis results file created at {args.analysis_output}")
        sys.exit(0) # Exit gracefully

    # --- Step 2: Convert and Save Scraped Data ---
    jobs_list = convert_and_save_scraped(jobs_df, args.scraped_jobs_file)
    if not jobs_list:
         log.error("Failed to convert or save scraped data. Pipeline halted.")
         sys.exit(1)

    # --- Step 3: Initialize Analyzer and Load Resume ---
    try:
        analyzer = ResumeAnalyzer()
    except Exception as e:
        log.critical(f"Failed to initialize ResumeAnalyzer: {e}. Cannot proceed with analysis.", exc_info=True)
        sys.exit(1)

    structured_resume = load_and_extract_resume(args.resume, analyzer)
    if not structured_resume:
        log.critical("Failed to load and extract data from resume. Pipeline halted.")
        sys.exit(1)

    # --- Step 4: Analyze Jobs ---
    # Note: analyze_jobs now uses rich progress bar internally if run from here
    analyzed_results = analyze_jobs(analyzer, structured_resume, jobs_list)
    if not analyzed_results:
         log.warning("Analysis step produced no results (all jobs might have failed analysis).")
         # Continue to save empty filtered list

    # --- Step 5: Apply Filters, Sort, and Save ---
    filter_args_dict = {}
    if args.min_salary is not None: filter_args_dict['salary_min'] = args.min_salary
    if args.max_salary is not None: filter_args_dict['salary_max'] = args.max_salary
    if args.filter_locations: filter_args_dict['locations'] = [loc.strip() for loc in args.filter_locations.split(',')]
    if args.filter_work_models: filter_args_dict['work_models'] = [wm.strip() for wm in args.filter_work_models.split(',')]
    if args.filter_job_types: filter_args_dict['job_types'] = [jt.strip() for jt in args.filter_job_types.split(',')]

    final_results_list_dict = apply_filters_sort_and_save(
        analyzed_results,
        args.analysis_output,
        filter_args_dict
    )

    # --- Step 6: Print Summary Table ---
    log.info("[bold blue]Pipeline Summary:[/bold blue]")
    print_summary_table(final_results_list_dict, top_n=10)

    log.info(f"[bold green]Pipeline Run Finished[/bold green]")

if __name__ == "__main__":
    # Ensure playwright browsers are installed before running
    # Consider adding a check here if needed
    run_pipeline()