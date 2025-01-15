import streamlit as st
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
import numpy as np
import pickle
from pathlib import Path


class ImageSearcher:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.database = None
        self.model = None
        self.processor = None

    @st.cache_resource
    def load_model(_self, model_name):
        model = CLIPModel.from_pretrained(model_name).to(_self.device)
        processor = CLIPProcessor.from_pretrained(model_name)
        return model, processor

    def load_database(self, database_path):
        with open(database_path, 'rb') as f:
            self.database = pickle.load(f)

        self.model, self.processor = self.load_model(self.database['model_name'])
        return len(self.database['image_paths'])

    @torch.no_grad()
    def search(self, query, threshold=0.2):
        text_inputs = self.processor(
            text=[query],
            return_tensors="pt",
            padding=True
        ).to(self.device)

        text_features = self.model.get_text_features(**text_inputs)
        text_features = text_features.cpu().numpy()

        text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)
        embeddings = self.database['embeddings'] / np.linalg.norm(self.database['embeddings'], axis=1, keepdims=True)

        similarities = np.dot(embeddings, text_features.T).squeeze()

        indices = np.argsort(similarities)[::-1]
        results = []

        for idx in indices:
            score = similarities[idx]
            if score < threshold:
                break

            results.append({
                'path': self.database['image_paths'][idx],
                'score': float(score)
            })

        return results


def main():
    st.set_page_config(layout="wide", page_title="Fast Image Searcher")
    st.title("Fast Image Search")

    if 'searcher' not in st.session_state:
        st.session_state.searcher = ImageSearcher()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_db_path = os.path.join(script_dir, 'image_embeddings.pkl')

    if 'database_loaded' not in st.session_state:
        st.session_state.database_loaded = False

    if not st.session_state.database_loaded:
        if os.path.exists(default_db_path):
            with st.spinner("Loading local database..."):
                try:
                    num_images = st.session_state.searcher.load_database(default_db_path)
                    st.session_state.database_loaded = True
                    st.success(f"Database loaded with {num_images} images")
                except Exception as e:
                    st.error(f"Error loading local database: {str(e)}")
        else:
            st.warning("Database file not found!")
            uploaded_file = st.file_uploader("Upload database file", type=['pkl'])
            if uploaded_file is not None:
                with st.spinner("Loading uploaded database..."):
                    try:
                        with open(default_db_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        num_images = st.session_state.searcher.load_database(default_db_path)
                        st.session_state.database_loaded = True
                        st.success(f"Database loaded with {num_images} images")
                    except Exception as e:
                        st.error(f"Error loading database: {str(e)}")

    if st.session_state.database_loaded:
        st.sidebar.header("Search Settings")
        threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.2)
        max_results = st.sidebar.slider("Maximum Results", 4, 500, 20, step=4)

        query = st.text_input("Enter your search query:")
        if query:
            with st.spinner("Searching..."):
                results = st.session_state.searcher.search(query, threshold)
                results = results[:max_results]

                if not results:
                    st.warning("No matching images found.")
                else:
                    st.success(f"Found {len(results)} matches")

                    for i in range(0, len(results), 4):
                        cols = st.columns(4)
                        batch = results[i:min(i + 4, len(results))]

                        for col, result in zip(cols, batch):
                            with col:
                                try:
                                    img_path = result['path']
                                    if os.path.exists(img_path):
                                        img = Image.open(img_path)
                                        st.image(img, use_container_width=True)
                                        st.write(f"Score: {result['score']:.3f}")
                                        width, height = img.size
                                        st.caption(f"Size: {width}x{height}")
                                        with st.expander("Path"):
                                            st.text(img_path)
                                    else:
                                        st.error("Image not found")
                                except Exception as e:
                                    st.error(f"Error loading image")


if __name__ == "__main__":
    main()