# precompute_features.py
"""
This script is an offline utility for the Visual Product Matcher application.

Its primary purpose is to perform the most time-consuming task—downloading
a dataset of images and extracting their feature vectors using a deep learning
model—and save the results to disk. This pre-computation step allows the main
Streamlit application to start almost instantly by loading the pre-computed
data, rather than re-processing the entire dataset every time it runs.

This script should be run once locally whenever the product dataset (metadata.csv)
is updated. The generated files (`features.npy` and `image_urls.pkl`) should then
be committed to the Git repository.
"""
import os
import pandas as pd
import requests
from PIL import Image, ImageOps
from io import BytesIO
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

# --- Configuration ---
# These file paths must match the constants in the main `image_search.py` app.
METADATA_FILE = "metadata.csv"
IMG_SIZE = (224, 224)
FEATURES_FILE = "features.npy"
IMAGE_URLS_FILE = "image_urls.pkl"

# ---------------------------------
# Core Processing Functions
# ---------------------------------

def load_feature_model():
    """Loads the pre-trained VGG16 model for feature extraction."""
    print("Loading VGG16 model...")
    model = keras.applications.vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    model.trainable = False
    print("Model loaded successfully.")
    return model

def load_metadata():
    """Loads the product metadata from the CSV file."""
    if not os.path.exists(METADATA_FILE):
        raise FileNotFoundError(f"Error: {METADATA_FILE} not found. Please ensure it's in the correct directory.")
    print(f"Loading metadata from {METADATA_FILE}...")
    return pd.read_csv(METADATA_FILE)

def preprocess_image(pil_img):
    """Preprocesses a single Pillow image for the VGG16 model."""
    img = ImageOps.fit(pil_img, IMG_SIZE, Image.Resampling.LANCZOS)
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = keras.applications.vgg16.preprocess_input(arr)
    return arr

def extract_features(pil_img, model):
    """Extracts a feature vector from an image."""
    processed_arr = preprocess_image(pil_img)
    # Set verbose=0 to prevent printing prediction progress for every single image.
    features = model.predict(processed_arr, verbose=0)
    return features.flatten()

# ---------------------------------
# Main Execution Block
# ---------------------------------

if __name__ == "__main__":
    # Step 1: Load the deep learning model and the product metadata.
    model = load_feature_model()
    metadata = load_metadata()

    # Ensure the required 'image_url' column exists in the CSV.
    if "image_url" not in metadata.columns:
        raise ValueError("Error: 'image_url' column not found in metadata.csv.")
        
    image_urls = metadata["image_url"].tolist()
    
    # Step 2: Iterate through all URLs, download images, and extract features.
    all_features = []
    processed_urls = []
    session = requests.Session()  # Use a session for potential connection pooling.
    
    print(f"Starting feature extraction for {len(image_urls)} images...")
    # Use tqdm to create a user-friendly progress bar in the terminal.
    for url in tqdm(image_urls, desc="Processing Images"):
        try:
            # Download the image with a timeout.
            response = session.get(url, timeout=15)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx).
            # Open the image from the response content.
            img = Image.open(BytesIO(response.content)).convert('RGB')
            # Extract features and store them.
            features = extract_features(img, model)
            all_features.append(features)
            processed_urls.append(url)
        except Exception as e:
            # Print a warning for any image that fails to process, then continue.
            print(f"\nWarning: Could not process URL: {url} | Reason: {e}")

    # Step 3: Save the collected features and corresponding URLs to disk.
    if not all_features:
        print("\nError: No features were extracted. Please check your image URLs and network connection.")
    else:
        # Convert the list of feature arrays into a single 2D NumPy matrix.
        feature_matrix = np.vstack(all_features)
        
        # Save the feature matrix as a .npy file for efficient loading.
        print(f"\nSaving feature matrix to {FEATURES_FILE}...")
        np.save(FEATURES_FILE, feature_matrix)
        
        # Save the list of successfully processed URLs using pickle.
        print(f"Saving processed image URLs to {IMAGE_URLS_FILE}...")
        with open(IMAGE_URLS_FILE, 'wb') as f:
            pickle.dump(processed_urls, f)
            
        print("\n✅ Pre-computation complete!")
        print(f"Successfully processed {len(processed_urls)} images.")
        print("You can now add 'features.npy' and 'image_urls.pkl' to your GitHub repository.")

