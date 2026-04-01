#!/usr/bin/env python3
"""
OCR Metadata Audit Pipeline for Streamlit
End-to-end image OCR, metadata extraction, structured field parsing, quality scoring
"""

import base64
import csv
import hashlib
import json
import math
import os
import re
import statistics
import struct
import textwrap
import unicodedata
from dataclasses import dataclass, field, asdict
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Image processing
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

# OCR
import pytesseract
from pytesseract import Output

# Output
import pandas as pd
import streamlit as st

# System
import exifread
import magic

# Tesseract path (adjust for your deployment)
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Linux/Streamlit Cloud

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# OCR
TESS_CONFIG_AUTO = "--oem 3 --psm 3"
TESS_CONFIG_SPARSE = "--oem 3 --psm 11"
TESS_LANG = "eng"
CONF_LOW = 60  # Token confidence threshold

# Scoring weights (must sum to 1.0)
WEIGHTS = {
    "ocr_confidence": 0.20,
    "metadata_confidence": 0.07,
    "completeness": 0.15,
    "legibility": 0.12,
    "layout": 0.08,
    "field_accuracy": 0.15,
    "ambiguity_risk": 0.08,
    "hallucination_risk": 0.07,
    "reproducibility": 0.08,
}

TRUST = {85: "High trust", 65: "Moderate trust", 45: "Low trust"}

# Regex field patterns
PATTERNS = {
    "dates": [
        re.compile(r"1[2-9]?\d{2}|[2-9]\d{3}", re.I),
        re.compile(r"1[2-9]?\d{2}st|nd|rd|th?", re.I),
        re.compile(r"\d{4}-\d{2}-\d{2}"),
    ],
    "emails": [re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}")],
    "phones": [re.compile(r"[\+]?[\d\s\-\(\)]{7,}")],
    "prices": [re.compile(r"[\£\$€]\d+[\.,]\d{2}")],
    "urls": [re.compile(r"https?://", re.I), re.compile(r"www\.", re.I)],
    "refs": [re.compile(r"[A-Z]{2,3}\d{2,3}[A-Z\-]?", re.I)],
}

SENSITIVITY_KW = [
    "confidential",
    "private",
    "restricted",
    "passport",
    "national insurance",
    "social security",
    "credit card",
    "medical",
    "classified",
]

REGION_LABELS = {
    (0, 0): "top-left",
    (0, 1): "top-centre",
    (0, 2): "top-right",
    (1, 0): "centre-left",
    (1, 1): "centre",
    (1, 2): "centre-right",
    (2, 0): "bottom-left",
    (2, 1): "bottom-centre",
    (2, 2): "bottom-right",
}

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class TechMeta:
    filename: str
    filepath: str
    size_bytes: int
    size_human: str
    ext: str
    mime: str
    width: int
    height: int
    aspect: str
    mode: str
    bit_depth: Optional[int]
    dpi_x: int
    dpi_y: int
    width_mm: Optional[float]
    height_mm: Optional[float]
    orientation: str
    is_grey: bool
    colours: List[str]
    exif: Dict
    gps: Optional[Dict]
    camera: Optional[str]
    software: Optional[str]
    created: Optional[str]
    modified: Optional[str]
    icc: Optional[str]
    has_metadata: bool


@dataclass
class OCRToken:
    text: str
    conf: int
    left: int
    top: int
    width: int
    height: int
    block: int
    line: int


@dataclass
class OCRResult;
