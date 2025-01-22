import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm


class ImageEmbeddingCreator:
    def __init__(self, model_name="openai/clip-vit-large-patch14-336"):
        print("Initializing CLIP model...")
        self.device = torch.device("cuda")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model_name = model_name
        print(f"Model loaded on {self.device}")

    @torch.no_grad()
    def create_image_embedding(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            image_features = self.model.get_image_features(**inputs)
            return image_features.cpu().numpy()
        except Exception as e:
            print(f"\nError processing {image_path}: {str(e)}")
            return None

    def create_database(self, image_directory, output_path, batch_size=32):
        # Get all image files
        print("Scanning for images...")
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        image_paths = [
            str(p) for p in Path(image_directory).rglob("*")
            if p.suffix.lower() in image_extensions
        ]

        if not image_paths:
            print(f"No images found in {image_directory}")
            return

        print(f"Found {len(image_paths)} images")

        # Process images and create embeddings
        embeddings = []
        valid_paths = []

        # Process in batches
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
            batch_paths = image_paths[i:i + batch_size]
            batch_embeddings = []

            for image_path in batch_paths:
                embedding = self.create_image_embedding(image_path)
                if embedding is not None:
                    batch_embeddings.append(embedding.squeeze())
                    valid_paths.append(image_path)

            if batch_embeddings:
                embeddings.extend(batch_embeddings)

        if not embeddings:
            print("No valid embeddings created")
            return

        print("\nCreating database...")
        database = {
            'model_name': self.model_name,
            'embeddings': np.stack(embeddings),
            'image_paths': valid_paths
        }

        print(f"Saving database to {output_path}...")
        with open(output_path, 'wb') as f:
            pickle.dump(database, f)

        print(f"Database created successfully with {len(valid_paths)} images")
        print(f"Embedding shape: {database['embeddings'].shape}")


def main():
    # Configuration
    image_directory = r"your-path-to-images"  # Replace with your image directory
    output_path = "image_embeddings.pkl"

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. This script requires a CUDA-capable GPU.")
        return

    # Create and save database
    creator = ImageEmbeddingCreator()
    creator.create_database(image_directory, output_path)


if __name__ == "__main__":
    main()