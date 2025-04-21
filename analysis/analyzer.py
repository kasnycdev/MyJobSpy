# Inside ResumeAnalyzer class

# --- extract_resume_data becomes ASYNC, adds TRUNCATION ---
async def extract_resume_data(self, resume_text: str) -> Optional[ResumeData]:
    """ASYNC Extracts structured data from resume text using the LLM."""
    # --- FIX: Ensure MAX_CHARS is an integer ---
    try:
        # Retrieve from config and explicitly convert to int
        max_chars_config = CFG.get('ollama', {}).get('max_prompt_chars', 24000)  # Get value (might be str or int)
        MAX_CHARS = int(max_chars_config)  # Convert to int
        log.debug(f"Using MAX_CHARS for resume truncation: {MAX_CHARS}")
    except (ValueError, TypeError) as e:
        log.error(
            f"Invalid value for ollama.max_prompt_chars in config: '{max_chars_config}'. Using default 24000. Error: {e}")
        MAX_CHARS = 24000  # Fallback to a default integer
    # --- END FIX ---

    if not resume_text or not resume_text.strip():
        log.warning("Resume text empty.");
        return None

    # --- Truncation Logic (Uses the now guaranteed integer MAX_CHARS) ---
    if len(resume_text) > MAX_CHARS:
        log.warning(f"Resume text ({len(resume_text)} chars) exceeds limit ({MAX_CHARS}). Truncating for extraction.")
        resume_text_for_prompt = resume_text[:MAX_CHARS]
    else:
        resume_text_for_prompt = resume_text
    # --- End Truncation ---

    prompt = self.resume_prompt_template.format(resume_text=resume_text_for_prompt)
    log.info("Requesting resume data extraction from LLM...")
    extracted_json = await self._call_ollama(prompt, request_context="resume extraction")

    if extracted_json:
        try:
            if isinstance(extracted_json, dict):
                resume_data = ResumeData(**extracted_json)
                log.info("Successfully parsed extracted resume data.")
                log.debug(
                    f"Extracted skills: T:{len(resume_data.technical_skills)} M:{len(resume_data.management_skills)}")
                log.debug(f"Extracted experience years: {resume_data.total_years_experience}")
                return resume_data
            else:
                log.error(f"LLM response for resume not dict: {type(extracted_json)}"); return None
        except Exception as e:  # Catch Pydantic validation errors too
            log.error(f"Failed to validate extracted resume data: {e}", exc_info=True);
            log.error(f"Invalid JSON received for resume: {extracted_json}");
            return None
    else:
        log.error("Failed to get valid JSON from LLM for resume extraction."); return None


# --- analyze_suitability method (Ensure similar int conversion if limits are used there) ---
async def analyze_suitability(self, resume_data: ResumeData, job_data: Dict[str, Any]) -> Optional[JobAnalysisResult]:
    # ... (previous code) ...
    try:
        # ... (prepare resume_data_json_str, temp_job_data, job_data_json_str) ...

        # --- FIX: Ensure prompt char limits from config are integers ---
        try:
            max_suitability_prompt_chars = int(CFG.get('ollama', {}).get('max_prompt_chars', 24000))
            max_job_desc_chars = int(CFG.get('ollama', {}).get('max_job_desc_chars_in_prompt',
                                                               5000))  # Assuming this was added to config too
        except (ValueError, TypeError) as e:
            log.error(f"Invalid integer value for Ollama char limits in config. Using defaults. Error: {e}")
            max_suitability_prompt_chars = 24000
            max_job_desc_chars = 5000
        # --- END FIX ---

        # --- Truncation Logic using integer limits ---
        PROMPT_TEMPLATE_OVERHEAD = 2000  # Keep as estimate
        resume_len = len(resume_data_json_str)
        job_len_no_desc = len(json.dumps({k: v for k, v in temp_job_data.items() if k != 'description'}, default=str))
        desc_len = len(job_desc)
        estimated_total_len = PROMPT_TEMPLATE_OVERHEAD + resume_len + job_len_no_desc + desc_len

        if estimated_total_len > max_suitability_prompt_chars:
            chars_over = estimated_total_len - max_suitability_prompt_chars
            keep_desc_len = max(0, desc_len - chars_over)
            final_desc_len = min(keep_desc_len, max_job_desc_chars)  # Use integer limit
            if final_desc_len < desc_len:
                log.warning(
                    f"Truncating job description for '{job_title}' from {desc_len} to {final_desc_len} chars...")
                temp_job_data["description"] = job_desc[:final_desc_len] + "..."
                job_data_json_str = json.dumps(temp_job_data, indent=2, default=str)
            else:
                log.warning(
                    f"Estimated prompt length ({estimated_total_len}) exceeds limit ({max_suitability_prompt_chars}) but desc already short/at max ({max_job_desc_chars}).")
                # Optionally still truncate desc if it exceeds its specific max len
                if desc_len > max_job_desc_chars:
                    log.warning(f"Truncating description to max allowed {max_job_desc_chars} chars anyway.")
                    temp_job_data["description"] = job_desc[:max_job_desc_chars] + "..."
                    job_data_json_str = json.dumps(temp_job_data, indent=2, default=str)

        prompt = self.suitability_prompt_template.format(
            resume_data_json=resume_data_json_str,
            job_data_json=job_data_json_str
        )
        # --- End Truncation Logic ---

    except Exception as e:
        log.error(f"Error preparing data for suitability prompt for '{job_title}': {e}", exc_info=True)
        return None

    # ... (rest of analyze_suitability unchanged: call ollama, process response) ...
    log.info(f"Requesting suitability analysis from LLM for job: {job_title}")
    combined_json_response = await self._call_ollama(prompt, request_context=f"suitability analysis for '{job_title}'")
    # ... (parsing and validation of response) ...
    if not combined_json_response or not isinstance(combined_json_response, dict): log.error(...); return None
    analysis_data = combined_json_response.get("analysis")
    if not analysis_data or not isinstance(analysis_data, dict): log.error(...); log.debug(...); return None
    try:
        analysis_result = JobAnalysisResult(**analysis_data);
        log.info(...);
        return analysis_result
    except Exception as e:
        log.error(...); log.error(...); return None


# --- close method unchanged ---
async def close(self):
    """Closes the async HTTP client."""
    if hasattr(self, 'async_client') and self.async_client:
        log.debug("Closing async Ollama client.")
        await self.async_client.aclose()