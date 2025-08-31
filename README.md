# ✨ Visual Product Matcher  
*An AI-Powered Visual Search Engine*  

---

## 🔴 Live Demo  

👉 [Check out the live application her](https://visual-appuct-matcher-jfoyitikgqfpk7yvaktmjy.streamlit.app/)  


---

## 📖 Project Overview  

The **Visual Product Matcher** is a sophisticated, AI-powered web application that allows users to find **visually similar products** from a catalog using an image.  

Instead of relying on ambiguous text-based searches, it leverages **deep learning** to analyze image content and return the closest matches within seconds.  

This project demonstrates a complete, end-to-end **Content-Based Image Retrieval (CBIR)** system with an optimized two-stage pipeline for **performance and scalability**.  

---

## 💡 Architecture & Problem-Solving Approach  

### ⚙️ Stage 1: Offline Pre-computation (Heavy Lifting)  
- **Script:** `precompute_features.py`  
- **Steps:**  
  - Read product catalog from `metadata.csv`  
  - Download each image  
  - Extract features using **VGG16** (pre-trained on ImageNet)  
  - Save results into:  
    - `features.npy` → pre-computed feature vectors  
    - `image_urls.pkl` → processed image URLs  

✅ This ensures the **live app loads instantly**, without recomputation.  

---

### ⚡ Stage 2: Real-time Similarity Search (Live App)  
- **Script:** `image_search.py`  
- **Steps:**  
  - Load pre-computed features on startup  
  - Build a **k-Nearest Neighbors (k-NN)** index with Scikit-learn  
  - Use **Cosine Similarity** for fast, robust comparisons  
  - On user input (uploaded image / URL):  
    - Extract features for query image  
    - Retrieve top matches from the index  
    - Display results in an **interactive Streamlit UI**  

---

## 🚀 Key Features  

- ⚡ **High-Performance Architecture** – Pre-computation ensures lightning-fast queries  
- 🧠 **Deep Learning Backend** – VGG16 feature extraction  
- 🖼️ **Dual Input Methods** – Upload file or paste image URL  
- 🎨 **Dynamic UI** – Responsive, modern Streamlit interface  
- ☁️ **Scalable Design** – Can handle large catalogs efficiently  

---

## 🛠️ Tech Stack  

- **Backend & Logic:** Python  
- **Framework:** Streamlit  
- **Deep Learning:** TensorFlow / Keras  
- **Indexing & ML:** Scikit-learn, NumPy  
- **Data Handling:** Pandas, Pickle  
- **Utilities:** Requests, Pillow, TQDM  

---

## 📂 Project Structure  

```bash
.
├── 📜 README.md # Documentation
├── 🐍 image_search.py # Main Streamlit app (fast loading)
├── ⚙️ precompute_features.py # Offline feature generation
├── 📋 metadata.csv # Product info + image URLs
├── 📦 requirements.txt # Dependencies
├── 🧠 features.npy # (Generated) Pre-computed vectors
└── 🔗 image_urls.pkl # (Generated) Processed URLs

```

---

## 🖥️ Running the Project Locally  

### 1️⃣ Clone the Repository 

```bash
git clone https://github.com/myselfshikhar/Visual-Product-Matcher.git
```
```bash
cd Visual-Product-Matcher
```


### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Pre-computation (One-Time Setup)
```bash
python precompute_features.py
```
(This may take several minutes depending on dataset size)

### 4️⃣ Start the Streamlit App
```bash
streamlit run image_search.py
```
👉 Now open http://localhost:8501 in your browser.

## 👨‍💻 Author
Shikhar Katiyar
