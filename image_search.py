# image_search.py
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageOps
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
import pandas as pd
import requests
from io import BytesIO

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Visual Product Matcher",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------
# Utilities & Backend Functions
# ---------------------------------
METADATA_FILE = "metadata.csv"
IMG_SIZE = (224, 224)

@st.cache_resource
def load_feature_model():
    """Loads the VGG16 model pre-trained for feature extraction."""
    model = keras.applications.vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    model.trainable = False
    return model

@st.cache_data(ttl=3600)
def load_metadata():
    """Loads the metadata CSV and returns a DataFrame."""
    if not os.path.exists(METADATA_FILE):
        return pd.DataFrame(columns=["id", "name", "category", "price", "image_url", "tags"])
    return pd.read_csv(METADATA_FILE)

def preprocess_image(pil_img):
    """Preprocesses a PIL image for VGG16 model input."""
    img = ImageOps.fit(pil_img, IMG_SIZE, Image.Resampling.LANCZOS)
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = keras.applications.vgg16.preprocess_input(arr)
    return arr

@st.cache_data(ttl=3600)
def extract_features(pil_img, model):
    """Extracts a feature vector from a processed image."""
    arr = preprocess_image(pil_img)
    features = model.predict(arr)
    return features.flatten()

@st.cache_data(ttl=3600)
def build_dataset_features(_metadata_df):
    """Downloads images from URLs and extracts features."""
    image_urls = _metadata_df["image_url"].tolist()
    if not image_urls:
        return np.array([]), []

    model = load_feature_model()
    all_features, processed_urls = [], []
    # Use a session for potential connection pooling
    session = requests.Session()
    progress_bar = st.progress(0, text="Processing dataset images...")
    for i, url in enumerate(image_urls):
        try:
            response = session.get(url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert('RGB')
            all_features.append(extract_features(img, model))
            processed_urls.append(url)
        except Exception as e:
            st.warning(f"Could not process image from URL: {url} | Error: {e}")
        progress_bar.progress((i + 1) / len(image_urls), text=f"Processing dataset... ({i+1}/{len(image_urls)})")
    progress_bar.empty()
    return np.vstack(all_features) if all_features else np.array([]), processed_urls

@st.cache_resource
def build_search_index(_features):
    """Builds the NearestNeighbors index."""
    if _features.shape[0] == 0:
        return None
    nn = NearestNeighbors(n_neighbors=20, metric='cosine', algorithm='brute')
    nn.fit(_features)
    return nn

# ---------------------------------
# UI Styling ("RECRUITER READY" THEME)
# ---------------------------------
def apply_custom_styling():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
        
        /* Core body styles */
        body {
            font-family: 'Poppins', sans-serif;
        }
        .stApp {
            background-color: #121212; /* Dark background */
            color: #EAEAEA; /* Light text for contrast */
        }
        
        /* Hide Streamlit's default elements */
        .st-emotion-cache-18ni7ap, .st-emotion-cache-h4xjwg { display: none; }
        
        /* Main Text colors */
        h1, h2, h3, h4, h5, h6, p, .stMarkdown { color: #EAEAEA !important; }

        /* Custom header with vibrant gradient */
        .header {
            padding: 1.5rem 1rem; /* Reduced padding */
            text-align: center;
            background: linear-gradient(135deg, #f857a6 0%, #ff5858 100%);
            border-radius: 12px;
            margin-bottom: 2rem;
        }
        .header h1 { font-size: 2.2rem; font-weight: 700; color: white !important; margin: 0; } /* Reduced font size */
        .header p { font-size: 1rem; color: white !important; opacity: 0.9; margin: 0.25rem 0 0 0; }

        /* Stylish upload section */
        .upload-section {
            background: #1E1E1E; /* Slightly lighter dark shade */
            padding: 2.5rem;
            border-radius: 12px;
            border: 1px solid #333;
            text-align: center;
            transition: all 0.3s ease-in-out;
        }
        .upload-section:hover {
            transform: translateY(-3px);
            box-shadow: 0 0 20px rgba(248, 87, 166, 0.3);
            border-color: #f857a6;
        }
        
        /* Main search button */
        .stButton>button {
            width: 100%;
            border: none;
            padding: 1rem;
            border-radius: 8px;
            background: linear-gradient(135deg, #f857a6 0%, #ff5858 100%);
            color: white;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease-in-out;
        }
        .stButton>button:hover {
            box-shadow: 0 0 25px #f857a6;
            transform: translateY(-3px);
        }
        
        /* Animation for result cards */
        @keyframes fadeIn {
            from { opacity: 0; transform: scale(0.95) translateY(10px); }
            to { opacity: 1; transform: scale(1) translateY(0); }
        }

        /* Result card styling with hover effect */
        .result-card {
            background: #1E1E1E;
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
            border: 1px solid #333;
            transition: all 0.3s ease-in-out;
            animation: fadeIn 0.5s ease-out;
        }
        .result-card:hover {
            transform: translateY(-5px);
            border-color: #f857a6;
            box-shadow: 0 6px 20px rgba(248, 87, 166, 0.15);
        }
        .st-emotion-cache-1kyxreqe {
             color: #EAEAEA; /* Progress bar text */
        }
        /* Placeholder for results area */
        .results-placeholder {
            background: #1E1E1E;
            padding: 4rem 2rem;
            border-radius: 12px;
            border: 2px dashed #333;
            text-align: center;
            transition: all 0.3s ease-in-out;
        }
        .results-placeholder p {
            color: #888 !important;
            font-size: 1.1rem;
        }
        .results-placeholder:hover {
            border-color: #f857a6;
            box-shadow: 0 0 20px rgba(248, 87, 166, 0.3);
        }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------
# UI Components
# ---------------------------------
def display_search_results(results, metadata_df):
    st.subheader(f"✨ Found {len(results)} similar items")
    if not results:
        st.info("No matches found. Try adjusting the similarity score.")
        return

    cols = st.columns(4)
    for i, r in enumerate(results):
        with cols[i % 4]:
            st.markdown(f"<div class='result-card'>", unsafe_allow_html=True)
            st.image(r['url'], use_container_width=True)
            meta_row = metadata_df[metadata_df.image_url == r['url']]
            if not meta_row.empty:
                meta = meta_row.iloc[0]
                st.markdown(f"**{meta.get('name', 'N/A')}**")
                price = meta.get('price', 0)
                try: price_str = f"₹{float(price):,.2f}"
                except (ValueError, TypeError): price_str = "N/A"
                st.markdown(f"<small style='color: #A9A9A9;'>{meta.get('category', 'N/A')} | **{price_str}**</small>", unsafe_allow_html=True)
            st.progress(r['score'], text=f"{r['score']:.0%}")
            st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------
# Main App Logic
# ---------------------------------

# Initialize session state to track if a search has been performed
if 'search_performed' not in st.session_state:
    st.session_state.search_performed = False

apply_custom_styling()

# -- Header --
st.markdown("""
<div class="header">
    <h1>Visual Product Matcher</h1>
    <p>AI-Powered Visual Search Engine</p>
</div>
""", unsafe_allow_html=True)

# -- Load Data (with spinner) --
with st.spinner("Initializing the search engine... Please wait."):
    metadata = load_metadata()
    if metadata.empty or "image_url" not in metadata.columns:
        st.error("Error: `metadata.csv` is missing or invalid.")
        st.stop()
    
    features, image_urls = build_dataset_features(metadata)
    if features.shape[0] == 0:
        st.error("Could not process any images. Please check your `metadata.csv` URLs.")
        search_index = None
    else:
        search_index = build_search_index(features)

# -- Main UI Layout --
col1, col_mid, col2 = st.columns([2, 1, 2])

with col1:
    st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
    st.subheader("Upload an Image")
    upload = st.file_uploader("Drag and drop or click to upload", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

with col_mid:
    st.markdown("<div style='text-align: center; padding-top: 4.5rem;'><p>OR</p></div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
    st.subheader("Paste an Image URL")
    url_input = st.text_input("Enter a direct link to an image", placeholder="https://...", label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

st.write("---")

# -- Preview and Search Area --
col_preview, col_search = st.columns([1, 1])

with col_preview:
    st.subheader("Image Preview")
    preview_slot = st.empty()

    input_image = None
    if upload:
        try: input_image = Image.open(upload).convert('RGB')
        except: st.error("Invalid uploaded file.")
    elif url_input:
        try:
            response = requests.get(url_input, timeout=10)
            response.raise_for_status()
            input_image = Image.open(BytesIO(response.content)).convert('RGB')
        except: st.error("Could not load image from URL.")
    
    if input_image:
        preview_slot.image(input_image, use_container_width=True)
    else:
        preview_slot.info("Your selected image will appear here.")

with col_search:
    st.subheader("Search Controls")
    top_k = st.slider("Max Results", 4, 20, 8, 4)
    min_score = st.slider("Minimum Similarity", 0.0, 1.0, 0.5, 0.05)
    search_btn = st.button("Search for Matches", use_container_width=True, type="primary")

# -- Results Area --
results_area = st.container()

# -- Search Action --
if search_btn:
    st.session_state.search_performed = True
    if input_image is None:
        st.error("Please upload or provide an image before searching.")
    elif search_index is None:
        st.error("Search engine is not available.")
    else:
        with st.spinner("Finding matches..."):
            model = load_feature_model()
            query_features = extract_features(input_image, model)
            dists, inds = search_index.kneighbors([query_features], n_neighbors=top_k)
            similarities = 1 - dists[0]
            
            results = []
            for sim, idx in zip(similarities, inds[0]):
                if sim >= min_score:
                    results.append({"score": float(sim), "url": image_urls[idx]})
            
            with results_area:
                display_search_results(results, metadata)

if not st.session_state.search_performed:
    with results_area:
        st.markdown("""
        <div class="results-placeholder">
            <p>Your search results will appear here ✨</p>
        </div>
        """, unsafe_allow_html=True)

