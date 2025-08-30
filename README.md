# ✨ Visual Product Matcher: An AI-Powered Visual Search Engine

---
<div align="center">

</div>

## 🔴 Live Demo
Check out the live application here:

[https://visual-appuct-matcher-jfoyitikgqfpk7yvaktmjy.streamlit.app/](https://visual-appuct-matcher-jfoyitikgqfpk7yvaktmjy.streamlit.app/)

---

## 📜 Project Overview
The Visual Product Matcher is a sophisticated, AI-powered web application that allows users to find visually similar products from a catalog using an image.

Instead of relying on text-based search, which can be ambiguous, this tool leverages the power of deep learning to understand the visual content of an image and retrieve the closest matches.

This project demonstrates a complete, end-to-end implementation of a content-based image retrieval (CBIR) system, from feature extraction with a pre-trained neural network to building a fast and efficient similarity search index. The entire application is wrapped in a sleek, interactive, and user-friendly web interface built with Streamlit.

---

## 🚀 Key Features

🖼️ Dual Input Methods: Users can either upload an image file directly or paste a URL to an image on the web.

🧠 Deep Learning Backend: Utilizes the powerful VGG16 convolutional neural network to extract high-level feature vectors from images.

⚡ Efficient Similarity Search: Employs a k-Nearest Neighbors (k-NN) algorithm with a cosine similarity metric to instantly find the most similar items.

🎨 Dynamic & Interactive UI: A modern and responsive user interface built with Streamlit, featuring real-time previews and adjustable search parameters.

☁️ Scalable Architecture: Designed to work with remote image URLs stored in a CSV, making it easy to scale the product catalog without local storage.

---

## 💡 Problem-Solving & Architecture

The core challenge is to quantify "visual similarity". My approach breaks this down into three main steps:

### 1️⃣ Feature Extraction (Image ➝ Vector)  

- An image is represented as a meaningful set of numbers (a feature vector) using the VGG16 model. This vector captures textures, patterns, shapes, and colors.

### 2️⃣ Fast Retrieval (Indexing)  

- To avoid slow, sequential searches, a search index is built using scikit-learn's NearestNeighbors. This pre-organizes all vectors for optimized, high-speed lookups.

- Cosine Similarity is used as the distance metric, as it excels at comparing high-dimensional vectors.

### 3️⃣ User Experience  

-Streamlit was selected to rapidly build a beautiful and intuitive Python-based web app, guiding the user seamlessly from input to result.

---

## 🛠️ Tech Stack  

- **Backend & Logic**: Python  
- **Web Framework**: Streamlit  
- **Deep Learning**: TensorFlow / Keras (VGG16)  
- **ML & Indexing**: Scikit-learn  
- **Data Handling**: Pandas, NumPy  
- **Image Processing**: Pillow  

---

## 📂 Project Structure  

```bash
.
├── 📜 README.md             # Project documentation  
├── 🐍 image_search.py       # Main Streamlit app  
├── 📋 metadata.csv          # Product info + image URLs  
└── 📦 requirements.txt      # Python dependencies
```

---

##🖥️ Running the Project Locally

Clone the repository:
```bash
git clone [https://github.com/myselfshikhar/Visual-Product-Matcher.git](https://github.com/myselfshikhar/Visual-Product-Matcher.git)
```
```bash
cd Visual-Product-Matcher
```


Install the required packages:
```bash
pip install -r requirements.txt
```
Run the Streamlit app:
```bash
streamlit run image_search.py
```
