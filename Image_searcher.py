import streamlit as st
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
import numpy as np
import pickle
from pathlib import Path


def process_directory(directory_path):
    valid_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    image_paths = []

    for path in Path(directory_path).rglob('*'):
        if path.suffix.lower() in valid_extensions:
            image_paths.append(str(path))

    return image_paths


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


def create_new_database():
    directory_path = st.text_input("Enter the full path to your images directory:")

    if directory_path and os.path.exists(directory_path):
        image_paths = process_directory(directory_path)

        if not image_paths:
            st.error("No valid images found in the directory")
            return False

        class EmbeddingBuilder:
            def __init__(self):
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model_name = "openai/clip-vit-large-patch14-336"
                self.image_size = 336

            def build(self, image_paths):
                # Create placeholder for progress
                progress_container = st.empty()
                with progress_container.container():
                    st.info("Loading CLIP model...")
                    model = CLIPModel.from_pretrained(self.model_name).to(self.device)
                    processor = CLIPProcessor.from_pretrained(self.model_name)

                    database = {
                        'embeddings': [],
                        'image_paths': [],
                        'model_name': self.model_name,
                        'image_size': self.image_size
                    }

                    total_images = len(image_paths)
                    st.info(f"Processing {total_images} images")

                    # Create the progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    progress_text = st.empty()

                    batch_size = 16
                    for i in range(0, total_images, batch_size):
                        batch_paths = image_paths[i:min(i + batch_size, total_images)]
                        batch_images = []
                        valid_paths = []

                        # Process batch
                        for path in batch_paths:
                            try:
                                image = Image.open(path)
                                if image.mode != 'RGB':
                                    image = image.convert('RGB')
                                image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
                                batch_images.append(image)
                                valid_paths.append(path)
                            except Exception as e:
                                st.error(f"Error processing {path}: {str(e)}")
                                continue

                        if batch_images:
                            with torch.no_grad():
                                inputs = processor(
                                    images=batch_images,
                                    return_tensors="pt",
                                    padding=True
                                ).to(self.device)

                                features = model.get_image_features(**inputs)
                                features = features.cpu().numpy()

                                database['embeddings'].extend(features)
                                database['image_paths'].extend(valid_paths)

                        # Update progress
                        current_progress = min((i + len(batch_paths)) / total_images, 1.0)
                        progress_bar.progress(current_progress)
                        progress_text.text(
                            f"Processing: {min(i + len(batch_paths), total_images)} / {total_images} images")
                        status_text.text(f"Current batch: {len(batch_paths)} images")

                    database['embeddings'] = np.array(database['embeddings'])

                    # Clean up progress display
                    progress_container.empty()
                    return database

        # Build database
        builder = EmbeddingBuilder()
        database = builder.build(image_paths)

        if database:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            save_path = os.path.join(script_dir, 'image_embeddings.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(database, f)
            st.success(f"Database created with {len(database['image_paths'])} images")
            return True

    return False


def main():
    st.set_page_config(layout="wide", page_title="Fast Image Searcher")
    st.title("Fast Image Search")

    if 'searcher' not in st.session_state:
        st.session_state.searcher = ImageSearcher()

    # Check for database file in script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_db_path = os.path.join(script_dir, 'image_embeddings.pkl')

    # Initialize database loading state
    if 'database_loaded' not in st.session_state:
        st.session_state.database_loaded = False

    # Try to load default database first
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
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Create New Database"):
                    if create_new_database():
                        with st.spinner("Loading newly created database..."):
                            num_images = st.session_state.searcher.load_database(default_db_path)
                            st.session_state.database_loaded = True
                            st.experimental_rerun()
            with col2:
                uploaded_file = st.file_uploader("Or upload existing database", type=['pkl'])
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

                    # Display 4 images per row
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