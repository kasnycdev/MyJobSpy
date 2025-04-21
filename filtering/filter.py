import logging
import time
import os
from typing import Dict, Optional, List, Any

# Import geopy and specific exceptions
try:
    from geopy.geocoders import Nominatim
    from geopy.distance import geodesic
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False

from .filter_utils import parse_salary, normalize_string
import config # Import config here for user agent

# Use root logger
log = logging.getLogger(__name__)

# --- Geocoding Setup ---
GEOCODER = None
if GEOPY_AVAILABLE:
    if not config.GEOPY_USER_AGENT or 'example.com' in config.GEOPY_USER_AGENT:
         log.warning("GEOPY_USER_AGENT in config.py is not set or uses placeholder. Geocoding might fail.")
    GEOCODER = Nominatim(user_agent=config.GEOPY_USER_AGENT, timeout=10)
else:
    log.warning("Geopy library not installed. Proximity and remote country filtering will be disabled.")

_geocode_cache = {}
_geocode_fail_cache = set()

# --- get_lat_lon_country function remains unchanged ---
def get_lat_lon_country(location_str: str) -> Optional[tuple[float, float, str]]:
    # (Content of this function remains the same as previous correct version)
    if not GEOPY_AVAILABLE or not GEOCODER or not location_str: return None
    normalized_loc = location_str.lower().strip();
    if not normalized_loc: return None
    if normalized_loc in _geocode_cache: return _geocode_cache[normalized_loc]
    if normalized_loc in _geocode_fail_cache: log.debug(f"Skipping geocode for previously failed: '{location_str}'"); return None
    log.debug(f"Geocoding location: '{location_str}'")
    try:
        time.sleep(1.0); location_data = GEOCODER.geocode(normalized_loc, addressdetails=True, language='en')
        if location_data and location_data.latitude and location_data.longitude:
            lat, lon = location_data.latitude, location_data.longitude; address = location_data.raw.get('address', {})
            country_code = address.get('country_code'); country_name = address.get('country')
            if country_code and not country_name:
                 cc_map = {'us': 'United States', 'usa': 'United States', 'ca': 'Canada', 'gb': 'United Kingdom', 'uk': 'United Kingdom', 'de': 'Germany', 'fr': 'France', 'au': 'Australia'}
                 country_name = cc_map.get(country_code.lower());
                 if country_name: log.debug(f"Inferred country name '{country_name}' from code '{country_code}'")
            if country_name:
                 log.debug(f"Geocoded '{location_str}' to ({lat:.4f}, {lon:.4f}), Country: {country_name}")
                 result = (lat, lon, country_name); _geocode_cache[normalized_loc] = result; return result
            else: log.warning(f"Geocoded '{location_str}' but couldn't extract country name. Address: {address}"); _geocode_fail_cache.add(normalized_loc); return None
        else: log.warning(f"Failed to geocode '{location_str}' - No results."); _geocode_fail_cache.add(normalized_loc); return None
    except (GeocoderTimedOut, GeocoderServiceError) as geo_err: log.error(f"Geocoding error for '{location_str}': {geo_err}"); _geocode_fail_cache.add(normalized_loc); return None
    except Exception as e: log.error(f"Unexpected geocoding error for '{location_str}': {e}", exc_info=True); _geocode_fail_cache.add(normalized_loc); return None


# --- Main Filter Function ---
def apply_filters(
    jobs: List[Dict[str, Any]],
    salary_min: Optional[int] = None, salary_max: Optional[int] = None,
    work_models: Optional[List[str]] = None, job_types: Optional[List[str]] = None,
    filter_remote_country: Optional[str] = None,
    filter_proximity_location: Optional[str] = None,
    filter_proximity_range: Optional[float] = None,
    filter_proximity_models: Optional[List[str]] = None,
    locations: Optional[List[str]] = None # Ignored
) -> List[Dict[str, Any]]:
    """ Filters jobs based on standard criteria PLUS advanced location filters. """
    filtered_jobs = []
    normalized_work_models = [normalize_string(wm) for wm in work_models] if work_models else []
    normalized_job_types = [normalize_string(jt) for jt in job_types] if job_types else []
    normalized_remote_country = normalize_string(filter_remote_country) if filter_remote_country else None
    normalized_proximity_models = [normalize_string(pm) for pm in filter_proximity_models] if filter_proximity_models else []

    target_lat_lon = None
    if filter_proximity_location and filter_proximity_range is not None:
        log.info(f"Attempting to geocode target proximity location: '{filter_proximity_location}'")
        target_geo_result = get_lat_lon_country(filter_proximity_location)
        if target_geo_result: target_lat_lon = (target_geo_result[0], target_geo_result[1]); log.info(f"Target proximity location geocoded to: {target_lat_lon}")
        else: log.error(f"Could not geocode target proximity location '{filter_proximity_location}'. Proximity filter disabled."); filter_proximity_location = None; filter_proximity_range = None

    log.info("Applying filters to job list...")
    initial_count = len(jobs)
    for job in jobs:
        job_title = job.get('title', 'N/A'); job_url = job.get('url', '#'); passes_all_filters = True

        # --- Standard Filters ---
        # Salary (Keep previous logic)
        job_min_salary, job_max_salary = None, None; salary_text = job.get('salary_text')
        if isinstance(salary_text, str) and salary_text: job_min_salary, job_max_salary = parse_salary(salary_text);
        if job_min_salary is not None or job_max_salary is not None: log.debug(f"Parsed salary for '{job_title}': Min={job_min_salary}, Max={job_max_salary} from '{salary_text}'")
        elif log.isEnabledFor(logging.DEBUG) and (salary_min is not None or salary_max is not None): log.debug(f"No salary info for '{job_title}' ({job_url})")
        salary_passes_check = True
        if salary_min is not None:
            if job_max_salary is not None:
                if job_max_salary < salary_min: salary_passes_check = False; log.debug(f"FILTERED (Salary Min): Max {job_max_salary} < filter {salary_min} for '{job_title}' ({job_url})")
            elif job_min_salary is not None:
                if job_min_salary < salary_min: salary_passes_check = False; log.debug(f"FILTERED (Salary Min): Min {job_min_salary} < filter {salary_min} for '{job_title}' ({job_url})")
            elif salary_passes_check and log.isEnabledFor(logging.DEBUG): log.debug(f"PASSED (Salary Min): No job salary data for '{job_title}'")
        if salary_passes_check and salary_max is not None:
            if job_min_salary is not None:
                if job_min_salary > salary_max: salary_passes_check = False; log.debug(f"FILTERED (Salary Max): Min {job_min_salary} > filter {salary_max} for '{job_title}' ({job_url})")
            elif job_max_salary is not None:
                if job_max_salary > salary_max: salary_passes_check = False; log.debug(f"FILTERED (Salary Max): Max {job_max_salary} > filter {salary_max} for '{job_title}' ({job_url})")
            elif salary_passes_check and log.isEnabledFor(logging.DEBUG): log.debug(f"PASSED (Salary Max): No job salary data for '{job_title}'")
        if not salary_passes_check: passes_all_filters = False; continue

        # Work Model (Standard) - CORRECTED BLOCK
        if normalized_work_models:
             job_model = normalize_string(job.get('work_model')) or normalize_string(job.get('remote'))
             if not job_model:
                  job_loc_wm = normalize_string(job.get('location'))
                  if 'remote' in job_loc_wm: job_model = 'remote'
                  elif 'hybrid' in job_loc_wm: job_model = 'hybrid'
                  elif 'on-site' in job_loc_wm or 'office' in job_loc_wm: job_model = 'on-site'
                  else: job_model = None
             if not job_model or job_model not in normalized_work_models:
                  passes_all_filters = False; log.debug(f"FILTERED (Work Model): '{job_title}' ({job_url}). Job model '{job_model}' not in {normalized_work_models}"); continue

        # Job Type
        if passes_all_filters and normalized_job_types:
            job_type_text = normalize_string(job.get('employment_type'))
            job_type_passes = False
            if job_type_text:
                 for jt_filter in normalized_job_types:
                      if jt_filter in job_type_text: job_type_passes = True; break
            if not job_type_passes: passes_all_filters = False; log.debug(f"FILTERED (Job Type): '{job_title}' ({job_url}). Type '{job_type_text}' not in {normalized_job_types}"); continue

        # --- Advanced Location Filters ---
        job_location_str = job.get('location', ''); job_geo_result = None
        # Filter 1: Remote Job in Specific Country
        if passes_all_filters and normalized_remote_country:
             job_model_rc = normalize_string(job.get('work_model')) or normalize_string(job.get('remote')); loc_text_rc = normalize_string(job_location_str); is_remote = job_model_rc == 'remote' or 'remote' in loc_text_rc
             if is_remote:
                  if not job_geo_result: job_geo_result = get_lat_lon_country(job_location_str)
                  job_country = job_geo_result[2] if job_geo_result else None
                  if not job_country or normalize_string(job_country) != normalized_remote_country: passes_all_filters = False; log.debug(f"FILTERED (Remote Country): Remote '{job_title}' ({job_url}). Country '{job_country}' != '{normalized_remote_country}'")
             else: passes_all_filters = False; log.debug(f"FILTERED (Remote Country): Job '{job_title}' ({job_url}) not remote, filter needs remote in '{normalized_remote_country}'")
             if not passes_all_filters: continue

        # Filter 2: Proximity
        if passes_all_filters and filter_proximity_location and target_lat_lon:
            job_model_prox = normalize_string(job.get('work_model')) or normalize_string(job.get('remote')); loc_text_prox = normalize_string(job_location_str)
            # --- CORRECTED INFERENCE BLOCK ---
            if not job_model_prox: # Infer if possible
                 if 'remote' in loc_text_prox:
                     job_model_prox = 'remote'
                 elif 'hybrid' in loc_text_prox:
                     job_model_prox = 'hybrid'
                 elif 'on-site' in loc_text_prox or 'office' in loc_text_prox:
                     job_model_prox = 'on-site'
                 else:
                     job_model_prox = None # Explicitly None if no match
            # --- END CORRECTION ---
            if not job_model_prox or job_model_prox not in normalized_proximity_models:
                 passes_all_filters = False
                 log.debug(f"FILTERED (Proximity Model): '{job_title}' ({job_url}). Model '{job_model_prox}' not allowed ({normalized_proximity_models}) for proximity filter.")
                 continue # Skip this job
            else:
                 # Proceed with geocoding and distance check only if model matches
                 if not job_geo_result: job_geo_result = get_lat_lon_country(job_location_str)
                 if not job_geo_result: passes_all_filters = False; log.debug(f"FILTERED (Proximity Geocode Fail): Could not geocode '{job_location_str}' for '{job_title}' ({job_url}).")
                 else:
                      job_lat_lon = (job_geo_result[0], job_geo_result[1])
                      try:
                          distance_miles = geodesic(target_lat_lon, job_lat_lon).miles
                          log.debug(f"Proximity check for '{job_title}' ({job_url}): Dist={distance_miles:.1f}mi from '{filter_proximity_location}'.")
                          if distance_miles > filter_proximity_range: passes_all_filters = False; log.debug(f"FILTERED (Proximity Range): Dist {distance_miles:.1f} > range {filter_proximity_range}mi for '{job_title}' ({job_url})")
                      except Exception as dist_err: # Catch potential errors in geodesic calculation
                           log.warning(f"Could not calculate distance between {target_lat_lon} and {job_lat_lon} for '{job_title}': {dist_err}")
                           passes_all_filters = False # Filter out if distance calculation fails
            if not passes_all_filters: continue

        # --- Final Decision ---
        if passes_all_filters: filtered_jobs.append(job)

    final_count = len(filtered_jobs)
    log.info(f"Filtering complete. {final_count} out of {initial_count} jobs passed all active filters.")
    return filtered_jobs