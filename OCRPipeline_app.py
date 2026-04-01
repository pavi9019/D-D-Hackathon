#!/usr/bin/env python3
"""
OCR Metadata Audit Pipeline for Streamlit
End-to-end image OCR, metadata extraction, structured field parsing, quality scoring
"""
#!/usr/bin/env python3
"""
OCR Metadata Audit Pipeline for Streamlit
Victorian Theatre Poster OCR + Metadata Extraction
"""

import base64
import csv
import json
import math
import os
import re
import statistics
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageEnhance
import pytesseract
from pytesseract import Output

# Tesseract path for Streamlit Cloud
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# ============================================================================
# CONFIG
# ============================================================================

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

TESS_LANG = "eng"
CONF_LOW = 60

# ============================================================================
# DATA MODELS (Fixed indentation)
# ============================================================================

@dataclass
class TechMeta:
    filename: str = ""
    width: int = 0
    height: int = 0
    size_human: str = ""
    dpi_x: int
