import io
import uvicorn
import numpy as np
import requests
from PIL import Image
from sklearn.cluster import KMeans
from typing import Tuple
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

origins = ["*"]



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials = True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_dominant_color(image_url: str) -> Tuple[int, int, int]:
    # Load the image from the URL
    response = requests.get(image_url)
    image = Image.open(io.BytesIO(response.content))
    # Load and preprocess the image
    image = image.resize((128, 128))  # Resize the image for faster processing
    image = image.convert("RGB")  # Convert to RGB format

    # Extract color features
    pixels = list(image.getdata())
    pixel_features = np.array([(r/255.0, g/255.0, b/255.0) for (r, g, b) in pixels])

    # Train or use a machine learning model
    kmeans = KMeans(n_clusters=5)  # Use K-means clustering with 5 clusters
    kmeans.fit(pixel_features)

    # Get the dominant color from the centroid of the largest cluster
    dominant_color = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]

    # Post-process the result
    # dominant_color_rgb = tuple((dominant_color * 255).astype(int))
    dominant_color_rgb = tuple(map(int, dominant_color * 255))
    return dominant_color_rgb

@app.get("/dominant-color")
async def get_dominant_color_api(image_url: str):
    dominant_color = get_dominant_color(image_url)
    return {"dominant_color": dominant_color}


