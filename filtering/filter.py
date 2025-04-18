import logging
import time
import os
from typing import Dict, Optional, List, Any
from functools import lru_cache # For caching geocode results

# Import geopy and specific exceptions
try:
    from geopy.geocoders import Nominatim
    from geopy.distance import geodesic
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False

from .filter_utils import parse_salary, normalize_string

# Use root logger configured by run_pipeline or main_matcher
log = logging.getLogger(__name__)

# --- Geocoding Setup ---
GEOCODER = None
if GEOPY_AVAILABLE:
    # IMPORTANT: Nominatim requires a unique user agent. Replace with your app name/email.
    APP_USER_AGENT = os.getenv("GEOPY_USER_AGENT", "MyJobSpyAnalysisApp/1.0 (myemail@example.com)")
    GEOCODER = Nominatim(user_agent=APP_USER_AGENT, timeout=10) # 10 second timeout
else:
    log.warning("Geopy library not installed. Proximity and remote country filtering will be disabled.")

# Cache geocoding results to avoid hitting the API repeatedly for the same location string
# Use a simple dictionary cache for this example
_geocode_cache = {}
_geocode_fail_cache = set() # Cache locations that failed geocoding

def get_lat_lon_country(location_str: str) -> Optional[tuple[float, float, str]]:
    """Geocodes a location string using Nominatim with caching and rate limiting."""
    if not GEOPY_AVAILABLE or not GEOCODER or not location_str:
        return None

    normalized_loc = location_str.lower().strip()
    if not normalized_loc:
         return None

    # Check cache first
    if normalized_loc in _geocode_cache:
        return _geocode_cache[normalized_loc]
    if normalized_loc in _geocode_fail_cache:
         log.debug(f"Skipping geocode for previously failed location: '{location_str}'")
         return None

    log.debug(f"Geocoding location: '{location_str}'")
    try:
        # --- Rate Limiting: Wait 1 second between requests to Nominatim ---
        time.sleep(1.0)
        location_data = GEOCODER.geocode(normalized_loc, addressdetails=True, language='en')

        if location_data and location_data.latitude and location_data.longitude:
            lat = location_data.latitude
            lon = location_data.longitude
            # Extract country code, then country name
            address = location_data.raw.get('address', {})
            country_code = address.get('country_code')
            country_name = address.get('country') # Get full country name
            if country_code and not country_name: # Fallback if full name missing
                 # Basic country code mapping (extend if needed)
                 cc_map = {'us': 'United States', 'ca': 'Canada', 'gb': 'United Kingdom'}
                 country_name = cc_map.get(country_code.lower())

            if country_name:
                 log.debug(f"Geocoded '{location_str}' to ({lat:.4f}, {lon:.4f}), Country: {country_name}")
                 result = (lat, lon, country_name)
                 _geocode_cache[normalized_loc] = result # Cache success
                 return result
            else:
                 log.warning(f"Geocoded '{location_str}' but couldn't extract country name. Address details: {address}")
                 _geocode_fail_cache.add(normalized_loc) # Cache failure
                 return None

        else:
            log.warning(f"Failed to geocode location: '{location_str}' - No results found.")
            _geocode_fail_cache.add(normalized_loc) # Cache failure
            return None

    except (GeocoderTimedOut, GeocoderServiceError) as geo_err:
        log.error(f"Geocoding error for '{location_str}': {geo_err}")
        _geocode_fail_cache.add(normalized_loc) # Cache failure (maybe temporary)
        return None
    except Exception as e:
        log.error(f"Unexpected error during geocoding for '{location_str}': {e}", exc_info=True)
        _geocode_fail_cache.add(normalized_loc) # Cache failure
        return None


# --- Main Filter Function ---
def apply_filters(
    jobs: List[Dict[str, Any]],
    # Standard Filters
    salary_min: Optional[int] = None,
    salary_max: Optional[int] = None,
    work_models: Optional[List[str]] = None, # General work model filter
    job_types: Optional[List[str]] = None,
    # Advanced Location Filters
    filter_remote_country: Optional[str] = None,
    filter_proximity_location: Optional[str] = None,
    filter_proximity_range: Optional[float] = None,
    filter_proximity_models: Optional[List[str]] = None,
    # Deprecated simple location filter (ignored now)
    locations: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Filters jobs based on standard criteria PLUS advanced location filters.
    """
    filtered_jobs = []
    # Normalize standard filters
    normalized_work_models = [normalize_string(wm) for wm in work_models] if work_models else []
    normalized_job_types = [normalize_string(jt) for jt in job_types] if job_types else []
    # Normalize advanced filters
    normalized_remote_country = normalize_string(filter_remote_country) if filter_remote_country else None
    normalized_proximity_models = [normalize_string(pm) for pm in filter_proximity_models] if filter_proximity_models else []

    # Pre-geocode target location for proximity filter
    target_lat_lon = None
    if filter_proximity_location and filter_proximity_range is not None:
        log.info(f"Attempting to geocode target proximity location: '{filter_proximity_location}'")
        target_geo_result = get_lat_lon_country(filter_proximity_location)
        if target_geo_result:
            target_lat_lon = (target_geo_result[0], target_geo_result[1])
            log.info(f"Target proximity location geocoded to: {target_lat_lon}")
        else:
            log.error(f"Could not geocode target proximity location '{filter_proximity_location}'. Proximity filter disabled.")
            # Disable proximity filter if target geocoding fails
            filter_proximity_location = None
            filter_proximity_range = None

    log.info("Applying filters to job list...")
    initial_count = len(jobs)
    for job in jobs:
        job_title = job.get('title', 'N/A')
        job_url = job.get('url', '#')
        passes_all_filters = True # Assume passes until a filter fails

        # --- Standard Filters (Apply First) ---

        # Salary (Unchanged logic from previous refined version)
        # ... (salary parsing and comparison logic remains here) ...
        job_min_salary, job_max_salary = None, None
        salary_text = job.get('salary_text')
        # ... rest of salary check ...
        # if not salary_passes: passes_all_filters = False; log.debug(...); continue

        # Work Model (Standard)
        if normalized_work_models:
             job_model = normalize_string(job.get('work_model')) or normalize_string(job.get('remote'))
             if not job_model: # Infer if possible
                  job_loc_wm = normalize_string(job.get('location'))
                  if 'remote' in job_loc_wm: job_model = 'remote'
                  elif 'hybrid' in job_loc_wm: job_model = 'hybrid'
                  elif 'on-site' in job_loc_wm or 'office' in job_loc_wm: job_model = 'on-site'

             if not job_model or job_model not in normalized_work_models:
                  passes_all_filters = False
                  log.debug(f"FILTERED (Work Model): '{job_title}' ({job_url}). Job model '{job_model}' not in {normalized_work_models}")

        # Job Type (Unchanged logic)
        if passes_all_filters and normalized_job_types:
            job_type_text = normalize_string(job.get('employment_type'))
            # ... rest of job type check ...
            # if not job_type_passes: passes_all_filters = False; log.debug(...);

        # --- Advanced Location Filters (Apply ONLY if standard filters passed) ---

        job_location_str = job.get('location', '')
        job_geo_result = None # Store geocoded result for job

        # Filter 1: Remote Job in Specific Country
        if passes_all_filters and normalized_remote_country:
             # Determine if job is remote
             job_model_rc = normalize_string(job.get('work_model')) or normalize_string(job.get('remote'))
             loc_text_rc = normalize_string(job_location_str)
             is_remote = job_model_rc == 'remote' or 'remote' in loc_text_rc

             if is_remote:
                  # Geocode job location to get reliable country
                  if not job_geo_result: # Geocode only if not already done
                       job_geo_result = get_lat_lon_country(job_location_str)

                  job_country = job_geo_result[2] if job_geo_result else None
                  if not job_country or normalize_string(job_country) != normalized_remote_country:
                       passes_all_filters = False
                       log.debug(f"FILTERED (Remote Country): Remote job '{job_title}' ({job_url}). Job country '{job_country}' != Filter country '{normalized_remote_country}'")
             else:
                  # If filter requires remote, but job isn't remote, filter it out
                  passes_all_filters = False
                  log.debug(f"FILTERED (Remote Country): Job '{job_title}' ({job_url}) is not remote, but filter requires remote in '{normalized_remote_country}'")

        # Filter 2: Proximity (Hybrid/On-site within Range)
        if passes_all_filters and filter_proximity_location and target_lat_lon:
            # Check if job work model is allowed for proximity
            job_model_prox = normalize_string(job.get('work_model')) or normalize_string(job.get('remote'))
            loc_text_prox = normalize_string(job_location_str)
            # Infer model if needed
            if not job_model_prox:
                if 'remote' in loc_text_prox: job_model_prox = 'remote'
                elif 'hybrid' in loc_text_prox: job_model_prox = 'hybrid'
                elif 'on-site' in loc_text_prox or 'office' in loc_text_prox: job_model_prox = 'on-site'

            if not job_model_prox or job_model_prox not in normalized_proximity_models:
                 passes_all_filters = False
                 log.debug(f"FILTERED (Proximity Model): Job '{job_title}' ({job_url}). Model '{job_model_prox}' not allowed ({normalized_proximity_models}) for proximity filter.")
            else:
                 # Geocode job location if not already done
                 if not job_geo_result:
                      job_geo_result = get_lat_lon_country(job_location_str)

                 if not job_geo_result:
                      passes_all_filters = False # Cannot check proximity if job location geocoding failed
                      log.debug(f"FILTERED (Proximity Geocode Fail): Could not geocode job location '{job_location_str}' for '{job_title}' ({job_url}).")
                 else:
                      job_lat_lon = (job_geo_result[0], job_geo_result[1])
                      # Calculate distance (ensure target_lat_lon is valid)
                      distance_miles = geodesic(target_lat_lon, job_lat_lon).miles
                      log.debug(f"Proximity check for '{job_title}' ({job_url}): Distance = {distance_miles:.1f} miles from '{filter_proximity_location}'.")

                      # Compare with range
                      if distance_miles > filter_proximity_range:
                           passes_all_filters = False
                           log.debug(f"FILTERED (Proximity Range): Job '{job_title}' ({job_url}) distance {distance_miles:.1f} > filter range {filter_proximity_range} miles.")


        # --- Final Decision ---
        if passes_all_filters:
            filtered_jobs.append(job)
        # else: # Already logged the specific reason for filtering above

    final_count = len(filtered_jobs)
    log.info(f"Filtering complete. {final_count} out of {initial_count} jobs passed all active filters.")
    return filtered_jobs