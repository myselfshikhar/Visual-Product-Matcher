# Image Similarity Search

This project implements a web-based image similarity search application using TensorFlow, Keras, scikit-learn, and Streamlit.

## Features

* **Image Upload:**  Users can upload a JPG image.
* **Feature Extraction:** Leverages a pre-trained VGG16 model (customizable) for extracting image features.
* **KNN-based Search:** Employs a KNeighborsClassifier model to find the most visually similar images within a dataset.
* **Streamlit UI:** Provides a user-friendly interface for interacting with the application.

## How to Run

1. **Prerequisites:**
    * Python 3.x
    * Install required libraries:
       ```bash
       pip install streamlit tensorflow keras numpy opencv-python pillow scikit-learn
       ```

2. **Get the Dataset:**
    * Place your image dataset in a folder named `dataset` within the project directory.  

3. **Start the Application:**
    ```bash
    streamlit run image_search.py  # Assuming your code is in 'image_search.py'
    ```

4. **Access in Browser:**  Open http://localhost:8501 (or the provided address) in your web browser.

## Optimizations

* **Efficient Feature Extractor:**  The project uses VGG16 for feature extraction. Consider experimenting with more lightweight models like MobileNetV2 for potential speed improvements.
* **Precomputed Features:**  For larger datasets, precalculate image features and store them for faster searches.
* **Approximate Nearest Neighbors (ANN):** Investigate ANN libraries like Faiss or Annoy to significantly speed up similarity searches, especially for very large datasets.

## Customization 

* **Feature Extractor:** Explore different pre-trained models in the `preprocess_image` function to experiment with accuracy and speed tradeoffs.
* **Dataset:** Replace the `dataset` folder with your own image collection.
* **Streamlit UI:**  Enhance the user interface with additional elements or styling using Streamlit's features.

## Contributing

Contributions are welcome! Feel free to open issues for suggestions or bug reports and submit pull requests to improve the project. 
