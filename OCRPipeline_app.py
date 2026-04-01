"""
Victorian Theatre Poster OCR Pipeline
Streamlit Cloud Deployment - NO OpenCV Required
"""

import base64
import io
import json
import streamlit as st
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pytesseract
import pandas as pd

# Tesseract config
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Config
TESS_CONFIG = "--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,:;-/()[] "
CONF_LOW = 50

# Page config
st.set_page_config(
    page_title="Theatre Poster OCR", 
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# APP UI
# ============================================================================

st.title("🎭 Victorian Theatre Poster OCR Pipeline")
st.markdown("**End-to-end OCR + metadata extraction for theatre posters**")

# Sidebar
st.sidebar.header("⚙️ Settings")
preprocess = st.sidebar.checkbox("Image preprocessing", value=True)
psm_mode = st.sidebar.selectbox("Page segmentation", ["6 (uniform block)", "3 (auto)", "11 (sparse text)"], index=0)
max_files = st.sidebar.slider("Max files", 1, 10, 3)

# File uploader
uploaded_files = st.file_uploader(
    "📁 Upload theatre posters (JPG/PNG)",
    type=['jpg', 'jpeg', 'png'],
    accept_multiple_files=True,
    max_uploaded_files=max_files
)

if uploaded_files:
    # Process images
    results = []
    
    for i, uploaded_file in enumerate(uploaded_files):
        if uploaded_file.type.startswith('image/'):
            # Load image
            image = Image.open(uploaded_file)
            st.subheader(f"📄 Processing {uploaded_file.name}")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Original
                st.image(image, caption="Original", use_column_width=True)
            
            with col2:
                # Process
                processed = preprocess_image(image) if preprocess else image
                ocr_data = run_ocr(processed, TESS_CONFIG.replace("psm 6", f"psm {psm_mode}"))
                
                # Display processed
                st.image(processed, caption="Processed", use_column_width=True)
                
                # OCR Results
                st.markdown("### 🔍 OCR Results")
                st.metric("Words found", ocr_data["word_count"])
                st.metric("Avg confidence", f"{ocr_data['avg_conf']:.1f}%")
                st.metric("High conf (>80%)", f"{ocr_data['high_conf_pct']:.1f}%")
                
                # Show text
                st.text_area("Extracted text", ocr_data["clean_text"], height=150)
                
                # Download JSON
                json_data = {
                    "filename": uploaded_file.name,
                    "timestamp": datetime.now().isoformat(),
                    "ocr": ocr_data,
                    "image_size": (image.width, image.height)
                }
                st.download_button(
                    label="💾 Download JSON",
                    data=json.dumps(json_data, indent=2),
                    file_name=f"{Path(uploaded_file.name).stem}_ocr.json",
                    mime="application/json"
                )
                
                results.append(json_data)
    
    # Batch summary
    if len(results) > 1:
        st.markdown("### 📊 Batch Summary")
        df = pd.DataFrame([{
            "File": r["filename"], 
            "Words": r["ocr"]["word_count"],
            "Avg Conf": f"{r['ocr']['avg_conf']:.1f}%",
            "High Conf": f"{r['ocr']['high_conf_pct']:.1f}%"
        } for r in results])
        st.dataframe(df)
        
        # CSV download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV Summary",
            data=csv,
            file_name="ocr_batch_summary.csv",
            mime="text/csv"
        )

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

@st.cache_data
def preprocess_image(img: Image.Image) -> Image.Image:
    """Basic PIL preprocessing for OCR"""
    steps = []
    
    # Convert to RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')
        steps.append("RGB")
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    steps.append("contrast")
    
    # Sharpen
    img = img.filter(ImageFilter.SHARPEN)
    steps.append("sharpen")
    
    return img

def run_ocr(img: Image.Image, config: str) -> Dict:
    """Run Tesseract OCR"""
    try:
        # Tesseract data
        data = pytesseract.image_to_data(
            img, 
            lang='eng',
            config=config,
            output_type=Output.DICT
        )
        
        # Parse results
        words = []
        confs = []
        clean_text = []
        
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            conf = float(data['conf'][i]) if data['conf'][i] != '-1' else 0
            
            if text and conf > 10:
                words.append(text)
                confs.append(conf)
                clean_text.append(text)
        
        # Stats
        avg_conf = statistics.mean(confs) if confs else 0
        high_conf_count = sum(1 for c in confs if c > 80)
        high_conf_pct = (high_conf_count / len(confs) * 100) if confs else 0
        
        return {
            "word_count": len(words),
            "avg_conf": avg_conf,
            "high_conf_pct": high_conf_pct,
            "clean_text": " ".join(clean_text),
            "raw_words": words,
            "confidence_scores": confs
        }
        
    except Exception as e:
        st.error(f"OCR failed: {str(e)}")
        return {
            "word_count": 0,
            "avg_conf": 0,
            "high_conf_pct": 0,
            "clean_text": "",
            "error": str(e)
        }

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        Built for theatre poster research<br>
        Powered by Tesseract OCR + Streamlit Cloud
    </div>
    """, 
    unsafe_allow_html=True
)
