import argparse
import json
import logging
import os
from datetime import datetime

from parsers.resume_parser import parse_resume
from parsers.job_parser import load_job_mandates
from analysis.analyzer import ResumeAnalyzer
from analysis.models import AnalyzedJob
from filtering.filter import apply_filters

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Analyze job mandates against a resume using GenAI.")
    parser.add_argument("--resume", required=True, help="Path to the resume file (.docx or .pdf)")
    parser.add_argument("--jobs", required=True, help="Path to the JSON file containing job mandates (list of objects)")
    parser.add_argument("--output", default="output/analyzed_jobs.json", help="Path for the output JSON file")

    # Filtering arguments
    parser.add_argument("--min-salary", type=int, help="Minimum desired annual salary (e.g., 80000)")
    parser.add_argument("--max-salary", type=int, help="Maximum desired annual salary (e.g., 120000)")
    parser.add_argument("--locations", help="Comma-separated list of desired locations (e.g., 'New York,Remote,London')")
    parser.add_argument("--work-models", help="Comma-separated list of desired work models (e.g., 'Remote,Hybrid')")
    parser.add_argument("--job-types", help="Comma-separated list of desired job types (e.g., 'Full-time,Contract')")

    args = parser.parse_args()

    # --- 1. Load Inputs ---
    logging.info("Starting job analysis process...")
    resume_text = parse_resume(args.resume)
    if not resume_text:
        logging.error("Failed to parse resume. Exiting.")
        return

    job_mandates = load_job_mandates(args.jobs)
    if not job_mandates:
        logging.error("Failed to load job mandates or file is empty. Exiting.")
        return

    # --- 2. Initialize Analyzer and Extract Resume Data ---
    try:
        analyzer = ResumeAnalyzer()
    except ConnectionError as e:
        logging.error(f"Could not initialize analyzer: {e}")
        return

    structured_resume_data = analyzer.extract_resume_data(resume_text)
    if not structured_resume_data:
        logging.error("Failed to extract structured data from resume. Cannot perform matching. Exiting.")
        return
    logging.info("Successfully extracted structured data from resume.")
    # Log extracted skills/experience years for verification
    logging.debug(f"Extracted Resume Skills: {structured_resume_data.skills}")
    logging.debug(f"Extracted Total Experience: {structured_resume_data.total_years_experience}")


    # --- 3. Apply Filters (Optional - BEFORE analysis for efficiency) ---
    filter_args = {}
    if args.min_salary is not None: filter_args['salary_min'] = args.min_salary
    if args.max_salary is not None: filter_args['salary_max'] = args.max_salary
    if args.locations: filter_args['locations'] = [loc.strip() for loc in args.locations.split(',')]
    if args.work_models: filter_args['work_models'] = [wm.strip() for wm in args.work_models.split(',')]
    if args.job_types: filter_args['job_types'] = [jt.strip() for jt in args.job_types.split(',')]

    if filter_args:
        logging.info("Applying pre-analysis filters...")
        eligible_jobs = apply_filters(job_mandates, **filter_args)
    else:
        eligible_jobs = job_mandates # No filters applied

    if not eligible_jobs:
        logging.warning("No jobs matched the specified filters. Exiting.")
        # Still save an empty list to the output file
        final_results_json = []
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(final_results_json, f, indent=4)
            logging.info(f"Empty results saved to {args.output}")
        except IOError as e:
            logging.error(f"Error writing empty results to output file {args.output}: {e}")
        return


    # --- 4. Analyze Filtered Jobs ---
    analyzed_results: list[AnalyzedJob] = []
    logging.info(f"Analyzing {len(eligible_jobs)} eligible job(s)...")
    for i, job in enumerate(eligible_jobs):
        logging.info(f"Analyzing job {i+1}/{len(eligible_jobs)}: {job.get('title', 'N/A')}")
        analysis = analyzer.analyze_suitability(structured_resume_data, job)

        if analysis:
            # Combine original job data with analysis results using Pydantic model
            analyzed_job = AnalyzedJob(original_job_data=job, analysis=analysis)
            analyzed_results.append(analyzed_job)
        else:
            logging.warning(f"Could not analyze job: {job.get('title', 'N/A')}. Skipping.")
            # Optionally include the job with a null analysis?
            # analyzed_job = AnalyzedJob(original_job_data=job, analysis=None) # Needs model adjustment
            # analyzed_results.append(analyzed_job)

    # --- 5. Sort Results ---
    logging.info("Sorting results by suitability score (descending)...")
    # Use model_dump() to get dicts for sorting if needed, or sort objects directly
    analyzed_results.sort(key=lambda x: x.analysis.suitability_score if x.analysis else -1, reverse=True)

    # --- 6. Save Output ---
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True) # Create output directory if it doesn't exist

    # Convert Pydantic models to dictionaries for JSON serialization
    final_results_json = [result.model_dump() for result in analyzed_results]

    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(final_results_json, f, indent=4)
        logging.info(f"Successfully saved {len(final_results_json)} analyzed jobs to {args.output}")
    except IOError as e:
        logging.error(f"Error writing output file {args.output}: {e}")
    except TypeError as e:
         logging.error(f"Error serializing results to JSON: {e}. Check data types in models.")
         logging.debug(f"Problematic data structure (first item): {final_results_json[0] if final_results_json else 'N/A'}")


if __name__ == "__main__":
    main()