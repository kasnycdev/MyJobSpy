# filtering/filter.py
import logging
import time
import os
import json
import atexit
from typing import Dict, Optional, List, Any
from datetime import datetime
from functools import lru_cache
from pathlib import Path

try:
    from geopy.geocoders import Nominatim
    from geopy.distance import geodesic
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False

from .filter_utils import parse_salary, normalize_string
from config import CFG, GEOCODE_CACHE_FILE # Use config for cache path & user agent

log = logging.getLogger(__name__)

# --- Geocoding Setup ---
_geocode_cache: Dict[str, Optional[tuple[float, float, str]]] = {}
_geocode_fail_cache: set[str] = set()
GEOCODER = None
_cache_loaded = False # Flag to prevent multiple loads/saves

def _load_geocode_cache():
    """Loads the geocode cache from disk if it exists."""
    global _geocode_cache, _geocode_fail_cache, _cache_loaded
    if _cache_loaded: return # Don't load twice
    if GEOCODE_CACHE_FILE.exists():
        log.debug(f"Loading geocode cache from {GEOCODE_CACHE_FILE}")
        try:
            with open(GEOCODE_CACHE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                _geocode_cache = data.get("success", {})
                _geocode_fail_cache = set(data.get("failure", []))
                log.info(f"Loaded {len(_geocode_cache)} success / {len(_geocode_fail_cache)} failure geocode entries.")
        except (json.JSONDecodeError, IOError, Exception) as e:
            log.error(f"Error loading geocode cache: {e}. Starting empty.")
            _geocode_cache = {}; _geocode_fail_cache = set()
    else:
        log.info("Geocode cache file not found. Starting empty.")
    _cache_loaded = True

def _save_geocode_cache():
    """Saves the geocode cache to disk."""
    if not _cache_loaded: return # Don't save if never loaded (or failed load badly)
    log.debug(f"Saving geocode cache to {GEOCODE_CACHE_FILE}")
    try:
        GEOCODE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        data_to_save = {"success": _geocode_cache, "failure": list(_geocode_fail_cache)}
        with open(GEOCODE_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2)
        log.info(f"Saved {len(_geocode_cache)} success / {len(_geocode_fail_cache)} failure geocode entries.")
    except Exception as e:
        log.error(f"Error saving geocode cache: {e}")

def initialize_geocoder():
    """Initializes the geocoder and loads cache."""
    global GEOCODER
    if GEOPY_AVAILABLE and not GEOCODER:
        user_agent = CFG['geocoding']['user_agent']
        if not user_agent or "your_email@example.com" in user_agent:
             log.warning("GEOPY_USER_AGENT not set properly. Geocoding may fail.")
        log.debug(f"Initializing Nominatim geocoder with user agent: {user_agent}")
        GEOCODER = Nominatim(user_agent=user_agent, timeout=10)
        _load_geocode_cache()
        atexit.register(_save_geocode_cache) # Register save on exit
    elif not GEOPY_AVAILABLE:
         log.warning("Geopy not installed. Location filtering disabled.")

# Initialize on module load
initialize_geocoder()

# --- get_lat_lon_country (Uses cache, saves failures) ---
# ... (Keep the get_lat_lon_country function as defined in the previous response) ...
def get_lat_lon_country(location_str: str) -> Optional[tuple[float, float, str]]:
    """Geocodes a location string using Nominatim with persistence cache."""
    if not GEOPY_AVAILABLE or not GEOCODER or not location_str: return None
    normalized_loc = location_str.lower().strip();
    if not normalized_loc: return None
    if normalized_loc in _geocode_cache: return _geocode_cache[normalized_loc]
    if normalized_loc in _geocode_fail_cache: log.debug(f"Skipping cached fail: '{location_str}'"); return None

    log.debug(f"Geocoding: '{location_str}'")
    try:
        time.sleep(1.0) # Rate limiting
        location_data = GEOCODER.geocode(normalized_loc, addressdetails=True, language='en')
        if location_data and location_data.latitude and location_data.longitude:
            lat, lon = location_data.latitude, location_data.longitude
            address = location_data.raw.get('address', {})
            country_code = address.get('country_code')
            country_name = address.get('country')
            if country_code and not country_name:
                 cc_map = {'us': 'United States', 'usa': 'United States', 'ca': 'Canada', 'gb': 'United Kingdom', 'uk': 'United Kingdom'}
                 country_name = cc_map.get(country_code.lower())
            if country_name:
                 result = (lat, lon, country_name); _geocode_cache[normalized_loc] = result; return result
            else: log.warning(f"Geocoded '{location_str}' but no country. Details: {address}"); _geocode_fail_cache.add(normalized_loc); return None
        else: log.warning(f"Failed geocode: '{location_str}' - No results."); _geocode_fail_cache.add(normalized_loc); return None
    except (GeocoderTimedOut, GeocoderServiceError) as geo_err: log.error(f"Geocoding error for '{location_str}': {geo_err}"); _geocode_fail_cache.add(normalized_loc); return None
    except Exception as e: log.error(f"Unexpected geocoding error for '{location_str}': {e}", exc_info=True); _geocode_fail_cache.add(normalized_loc); return None


# --- apply_filters function (Updated with new filters) ---
def apply_filters(
    jobs: List[Dict[str, Any]],
    # Standard Filters
    salary_min: Optional[int] = None, salary_max: Optional[int] = None,
    work_models: Optional[List[str]] = None, job_types: Optional[List[str]] = None,
    # NEW Filters
    filter_companies: Optional[List[str]] = None,
    exclude_companies: Optional[List[str]] = None,
    filter_title_keywords: Optional[List[str]] = None,
    filter_date_after: Optional[str] = None,
    filter_date_before: Optional[str] = None,
    # Advanced Location Filters
    filter_remote_country: Optional[str] = None,
    filter_proximity_location: Optional[str] = None,
    filter_proximity_range: Optional[float] = None,
    filter_proximity_models: Optional[List[str]] = None,
    # Min score filter is applied separately in main_matcher after analysis
) -> List[Dict[str, Any]]:
    """Filters jobs based on various criteria BEFORE analysis."""
    # Disable location filters if geopy not available
    if not GEOPY_AVAILABLE and (filter_remote_country or filter_proximity_location):
         log.error("Geopy unavailable, cannot perform advanced location filtering.")
         filter_remote_country = None; filter_proximity_location = None

    # Normalize filters once
    normalized_work_models = {normalize_string(wm) for wm in work_models} if work_models else set()
    normalized_job_types = {normalize_string(jt) for jt in job_types} if job_types else set()
    normalized_remote_country = normalize_string(filter_remote_country) if filter_remote_country else None
    normalized_proximity_models = {normalize_string(pm) for pm in filter_proximity_models} if filter_proximity_models else set()
    # New filter normalization
    normalized_include_companies = {normalize_string(c) for c in filter_companies} if filter_companies else set()
    normalized_exclude_companies = {normalize_string(c) for c in exclude_companies} if exclude_companies else set()
    normalized_title_keywords = {normalize_string(k) for k in filter_title_keywords} if filter_title_keywords else set()
    date_after = None
    date_before = None
    try: date_after = datetime.strptime(filter_date_after, '%Y-%m-%d').date() if filter_date_after else None
    except ValueError: log.warning(f"Invalid --filter-date-after format: {filter_date_after}. Use YYYY-MM-DD."); date_after = None
    try: date_before = datetime.strptime(filter_date_before, '%Y-%m-%d').date() if filter_date_before else None
    except ValueError: log.warning(f"Invalid --filter-date-before format: {filter_date_before}. Use YYYY-MM-DD."); date_before = None

    # Pre-geocode target location
    target_lat_lon = None
    if filter_proximity_location and filter_proximity_range is not None:
        # ... (keep target geocoding logic as before) ...
        target_geo_result = get_lat_lon_country(filter_proximity_location)
        if target_geo_result: target_lat_lon = (target_geo_result[0], target_geo_result[1]); log.info(f"Target geocoded: {target_lat_lon}")
        else: log.error(f"Could not geocode proximity target '{filter_proximity_location}'. Proximity filter disabled."); filter_proximity_location = None


    log.info("Applying pre-analysis filters to job list...")
    initial_count = len(jobs)
    filtered_jobs = []
    for job in jobs:
        job_title = job.get('title', 'N/A')
        job_url = job.get('url', '#')
        passes_all_filters = True

        # --- Filter Logic ---

        # Salary Filter
        # ... (keep salary filter logic) ...

        # Work Model (Standard)
        if passes_all_filters and normalized_work_models:
             # ... (keep work model logic, compare against normalized_work_models set) ...
             job_model = normalize_string(job.get('work_model')) # ... infer if needed ...
             if not job_model or job_model not in normalized_work_models: passes_all_filters = False; log.debug(...)

        # Job Type
        if passes_all_filters and normalized_job_types:
             # ... (keep job type logic, compare against normalized_job_types set) ...
             job_type_text = normalize_string(job.get('employment_type'))
             if not job_type_text or job_type_text not in normalized_job_types: passes_all_filters = False; log.debug(...)

        # --- NEW Filters ---
        # Company Include Filter
        if passes_all_filters and normalized_include_companies:
            company_name = normalize_string(job.get('company'))
            if not company_name or company_name not in normalized_include_companies:
                passes_all_filters = False
                log.debug(f"FILTERED (Company Include): '{job_title}' ({job_url}). Company '{job.get('company')}' not in inclusion list.")

        # Company Exclude Filter
        if passes_all_filters and normalized_exclude_companies:
            company_name = normalize_string(job.get('company'))
            if company_name and company_name in normalized_exclude_companies:
                passes_all_filters = False
                log.debug(f"FILTERED (Company Exclude): '{job_title}' ({job_url}). Company '{job.get('company')}' is in exclusion list.")

        # Title Keywords Filter (ANY keyword must match)
        if passes_all_filters and normalized_title_keywords:
            title_norm = normalize_string(job_title)
            if not any(keyword in title_norm for keyword in normalized_title_keywords):
                 passes_all_filters = False
                 log.debug(f"FILTERED (Title Keywords): '{job_title}' ({job_url}). Title does not contain any of {normalized_title_keywords}.")

        # Date Posted Filter
        if passes_all_filters and (date_after or date_before):
             job_date_str = job.get('date_posted') # Assumes YYYY-MM-DD format after conversion
             job_date = None
             try: job_date = datetime.strptime(job_date_str, '%Y-%m-%d').date() if job_date_str else None
             except ValueError: log.warning(f"Could not parse job date '{job_date_str}' for '{job_title}'. Skipping date filter."); job_date=None

             if job_date:
                 if date_after and job_date < date_after:
                      passes_all_filters = False
                      log.debug(f"FILTERED (Date Posted): '{job_title}' ({job_url}). Date {job_date} is before {date_after}.")
                 if date_before and job_date > date_before:
                      passes_all_filters = False
                      log.debug(f"FILTERED (Date Posted): '{job_title}' ({job_url}). Date {job_date} is after {date_before}.")
             elif date_after or date_before: # If filtering by date but job has no valid date
                  passes_all_filters = False
                  log.debug(f"FILTERED (Date Posted): '{job_title}' ({job_url}). Missing valid post date for comparison.")
        # --- End NEW Filters ---


        # --- Advanced Location Filters ---
        job_location_str = job.get('location', '')
        job_geo_result = None

        # Remote Country
        if passes_all_filters and normalized_remote_country:
             # ... (keep remote country logic using get_lat_lon_country) ...
             job_model_rc = normalize_string(job.get('work_model')) # ... infer if needed ...
             is_remote = job_model_rc == 'remote' or 'remote' in normalize_string(job_location_str)
             if is_remote:
                  job_geo_result = get_lat_lon_country(job_location_str)
                  job_country = job_geo_result[2] if job_geo_result else None
                  if not job_country or normalize_string(job_country) != normalized_remote_country: passes_all_filters = False; log.debug(...)
             else: passes_all_filters = False; log.debug(...)

        # Proximity
        if passes_all_filters and filter_proximity_location and target_lat_lon:
             # ... (keep proximity logic using get_lat_lon_country and normalized_proximity_models) ...
             job_model_prox = normalize_string(job.get('work_model')) # ... infer if needed ...
             if not job_model_prox or job_model_prox not in normalized_proximity_models: passes_all_filters = False; log.debug(...)
             else:
                  if not job_geo_result: job_geo_result = get_lat_lon_country(job_location_str)
                  if not job_geo_result: passes_all_filters = False; log.debug(...)
                  else:
                       job_lat_lon = (job_geo_result[0], job_geo_result[1])
                       distance_miles = geodesic(target_lat_lon, job_lat_lon).miles
                       log.debug(f"Proximity check: '{job_title}' Dist={distance_miles:.1f} mi.")
                       if distance_miles > filter_proximity_range: passes_all_filters = False; log.debug(...)

        # --- Final Decision ---
        if passes_all_filters:
            filtered_jobs.append(job)

    final_count = len(filtered_jobs)
    log.info(f"Filtering complete. {final_count} out of {initial_count} jobs passed pre-analysis filters.")
    return filtered_jobs