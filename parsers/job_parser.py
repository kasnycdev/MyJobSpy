import json
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_job_mandates(json_file_path: str) -> list[dict]:
    """
    Loads job mandates from a JSON file.

    Args:
        json_file_path: Path to the JSON file containing a list of job objects.

    Returns:
        A list of job dictionaries, or an empty list on error.
    """
    if not os.path.exists(json_file_path):
        logging.error(f"Job mandates file not found: {json_file_path}")
        return []

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            logging.info(f"Successfully loaded {len(data)} job mandates from {json_file_path}")
            # Basic validation: ensure items are dictionaries
            if all(isinstance(job, dict) for job in data):
                return data
            else:
                logging.error("JSON file does not contain a list of job objects (dictionaries).")
                return []
        else:
            logging.error("JSON file root is not a list.")
            return []
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON file {json_file_path}: {e}")
        return []
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading {json_file_path}: {e}")
        return []