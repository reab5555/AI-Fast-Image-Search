# AI Fast Image Search

This AI based tool is made for managing and searching large image collections seamlessly with machine learning-based text query capabilities.

Fast Image Searcher offers an advanced solution for efficiently organizing, querying, and retrieving images from large-scale collections. Leveraging AI/machine learning, this tool enables precise and swift text-based image searches.

## Features

- **Efficient Image Database Creation**: Generate a compact representation (embedding) of each image in your collection using the CLIP model. This transforms the images into a space where similar items are close together, facilitating rapid searches.
  
- **Search with Text Queries**: Input a search query or question, and the tool will return the most relevant images from your database. This is powered by the CLIP model's ability to understand and compare text and image semantics.

- **Rapid Lookup**: The tool uses vector similarity search for finding the images, ensuring quick responses even with a large database.

- **Threshold-Based Filtering**: Customize the sensitivity of your search results with a similarity threshold.

- **Results Customization**: Limit the maximum number of results displayed, and view images in an easy-to-navigate grid format.

## Installation

The tool requires Python to run. Clone the repository and install the dependencies:

```bash
git clone https://github.com/reab5555/AI-Fast-Image-Search.git
cd AI-Fast-Image-Search
pip install -r requirements.txt
```

## Usage

1. **Create a Database**:
   - First, you can either upload an existing database file or create a new one by pointing to a directory of images.

2. **Search Images**:
   - Enter your search query in the provided input box. Adjust the similarity threshold and maximum results as needed in the sidebar.

3. **View Results**: 
   - The search results will display images that best match your query, along with their similarity scores.

## Getting Started

To run the application locally, execute the following command:

```bash
streamlit run app.py
```

Open your browser and navigate to the provided local URL to access the interface.
