import os
from docx import Document
import PyPDF2
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _parse_docx(file_path: str) -> str:
    """Parses text content from a DOCX file."""
    try:
        doc = Document(file_path)
        full_text = [para.text for para in doc.paragraphs]
        return '\n'.join(full_text)
    except Exception as e:
        logging.error(f"Error parsing DOCX file {file_path}: {e}")
        return ""

def _parse_pdf(file_path: str) -> str:
    """Parses text content from a PDF file using PyPDF2."""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        logging.error(f"Error parsing PDF file {file_path}: {e}")
        # Consider adding pdfminer.six as a fallback here if needed
        return ""

def parse_resume(file_path: str) -> str:
    """
    Parses text from a resume file (DOCX or PDF).

    Args:
        file_path: Path to the resume file.

    Returns:
        The extracted text content as a string, or empty string on error.
    """
    if not os.path.exists(file_path):
        logging.error(f"Resume file not found: {file_path}")
        return ""

    _, file_extension = os.path.splitext(file_path.lower())

    if file_extension == ".docx":
        logging.info(f"Parsing DOCX resume: {file_path}")
        return _parse_docx(file_path)
    elif file_extension == ".pdf":
        logging.info(f"Parsing PDF resume: {file_path}")
        return _parse_pdf(file_path)
    else:
        logging.error(f"Unsupported resume file format: {file_extension}. Please use .docx or .pdf.")
        return ""